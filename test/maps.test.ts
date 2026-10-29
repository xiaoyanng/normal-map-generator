import { describe, expect, test } from 'vitest';
import type { RgbImage } from '../src/image';
import {
  computeMaps,
  DEFAULT_SETTINGS,
  heightMapScore,
  type MapSet,
  type MapSettings,
} from '../src/maps';

/**
 * Image mode on a height map: a smooth dome (white top, black floor) must
 * come out as a dome, its sides tilting away from the middle. The dome is a
 * large shape, the band the "remove uneven lighting" switch drops.
 */
const SIZE = 256;
const RADIUS = 100;
const C = SIZE / 2;

function dome(): RgbImage {
  const plane = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const r2 = (x - C) ** 2 + (y - C) ** 2;
      plane[y * SIZE + x] = Math.sqrt(Math.max(0, RADIUS ** 2 - r2)) / RADIUS;
    }
  }
  return { width: SIZE, height: SIZE, r: plane, g: plane, b: plane };
}

const imageMode = (patch: Partial<MapSettings> = {}): MapSet =>
  computeMaps(dome(), {
    ...DEFAULT_SETTINGS,
    source: 'image',
    ...patch,
  });

/** Encoded normal (0..1) at a pixel, as a -1..1 vector. */
function normalAt(maps: MapSet, x: number, y: number) {
  const i = y * SIZE + x;
  return maps.normal.map((plane) => plane[i] * 2 - 1);
}

/** Mean sideways tilt over the inner half of the dome (rim excluded). */
function innerTilt(maps: MapSet): number {
  let sum = 0;
  let count = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const r = Math.hypot(x - C, y - C);
      if (r < RADIUS * 0.3 || r > RADIUS * 0.8) continue;
      const [nx, ny] = normalAt(maps, x, y);
      sum += Math.hypot(nx, ny);
      count++;
    }
  }
  return sum / count;
}

describe('normal map engine, image mode', () => {
  test('a dome height map keeps its shape by default', () => {
    const maps = imageMode();
    const d = 60;
    // OpenGL: x right, y up. Each side faces away from the middle.
    expect(normalAt(maps, C - d, C)[0]).toBeLessThan(-0.1);
    expect(normalAt(maps, C + d, C)[0]).toBeGreaterThan(0.1);
    expect(normalAt(maps, C, C - d)[1]).toBeGreaterThan(0.1);
    expect(normalAt(maps, C, C + d)[1]).toBeLessThan(-0.1);
    // Like a hemisphere, the tilt grows from the top towards the rim.
    const tilt = [0.3, 0.6, 0.9].map(
      (f) => -normalAt(maps, C - Math.round(RADIUS * f), C)[0]
    );
    expect(tilt[0]).toBeGreaterThan(0.03);
    expect(tilt[1]).toBeGreaterThan(tilt[0] * 1.5);
    expect(tilt[2]).toBeGreaterThan(tilt[1] * 1.5);
  });

  test('halfway up the dome the normal clearly points outwards', () => {
    const maps = imageMode();
    const byte = (v: number) => Math.round((v * 0.5 + 0.5) * 255);
    const r = RADIUS * 0.5;
    // On the axes (one channel tilts) and on the diagonal (both tilt).
    const left = normalAt(maps, C - r, C);
    const top = normalAt(maps, C, C - r);
    const d = Math.round(r / Math.SQRT2);
    const diagonal = normalAt(maps, C - d, C - d);
    expect(Math.abs(byte(left[0]) - 128)).toBeGreaterThanOrEqual(20);
    expect(Math.abs(byte(top[1]) - 128)).toBeGreaterThanOrEqual(20);
    expect(Math.abs(byte(diagonal[0]) - 128)).toBeGreaterThanOrEqual(20);
    expect(Math.abs(byte(diagonal[1]) - 128)).toBeGreaterThanOrEqual(20);
  });

  test('DirectX flips only the green channel', () => {
    const gl = imageMode();
    const dx = imageMode({ format: 'directx' });
    const [gx, gy] = normalAt(gl, C, C - 60);
    const [xx, xy] = normalAt(dx, C, C - 60);
    expect(xx).toBeCloseTo(gx, 5);
    expect(xy).toBeCloseTo(-gy, 5);
  });

  test('removing uneven lighting drops the large shape', () => {
    const kept = innerTilt(imageMode());
    const flattened = innerTilt(imageMode({ flattenLighting: true }));
    expect(kept).toBeGreaterThan(0.05);
    expect(flattened).toBeLessThan(kept * 0.25);
  });
});

/**
 * Steep-rimmed hemispheres saved as an 8-bit gray PNG would be:
 * 40 + 200 * sqrt(1 - r²), 40 outside, `count` x `count` of them.
 */
function hemispheres(size: number, count: number, fill: number) {
  const cell = size / count;
  const radius = cell * fill;
  const plane = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cx = (Math.floor(x / cell) + 0.5) * cell;
      const cy = (Math.floor(y / cell) + 0.5) * cell;
      const r = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
      plane[y * size + x] =
        Math.round(r > 1 ? 40 : 40 + 200 * Math.sqrt(1 - r * r)) / 255;
    }
  }
  const image: RgbImage = {
    width: size,
    height: size,
    r: plane,
    g: plane,
    b: plane,
  };
  return { image, radius, cell };
}

/** Byte offsets from 128 of R and G halfway up the first hemisphere. */
function tiltAtHalf(size: number, count: number, fill: number) {
  const { image, radius, cell } = hemispheres(size, count, fill);
  const maps = computeMaps(image, { ...DEFAULT_SETTINGS, source: 'image' });
  const c = cell / 2;
  const d = Math.round(radius / 2);
  const byte = (plane: Float32Array, x: number, y: number) =>
    Math.round(plane[Math.round(y) * size + Math.round(x)] * 255);
  const diag = Math.round(radius / 2 / Math.SQRT2);
  return {
    left: Math.abs(byte(maps.normal[0], c - d, c) - 128),
    top: Math.abs(byte(maps.normal[1], c, c - d) - 128),
    diagonal: [
      Math.abs(byte(maps.normal[0], c - diag, c - diag) - 128),
      Math.abs(byte(maps.normal[1], c - diag, c - diag) - 128),
    ],
  };
}

describe('normal map engine, steep height maps', () => {
  test('a steep-rimmed hemisphere tilts clearly halfway up', () => {
    const tilt = tiltAtHalf(256, 1, 0.39);
    expect(tilt.left).toBeGreaterThanOrEqual(20);
    expect(tilt.top).toBeGreaterThanOrEqual(20);
    // On the diagonal the tilt splits between R and G (each about the axis
    // value over sqrt 2); together they lean as far.
    expect(Math.hypot(...tilt.diagonal)).toBeGreaterThanOrEqual(20);
  });

  test('many steep hemispheres keep their real slopes (no photo cap)', () => {
    // Their rims make the photo scale small; capping a clean height map at
    // three times it gave about 37 here, the real slopes give about 75.
    const tilt = tiltAtHalf(256, 4, 0.45);
    expect(tilt.left).toBeGreaterThanOrEqual(60);
    expect(tilt.top).toBeGreaterThanOrEqual(60);
  });

  test('a large 8-bit hemisphere does not flatten between gray levels', () => {
    // At 2048 px a gentle slope climbs one gray level every few pixels;
    // slopes read straight from the steps came out exactly flat.
    const tilt = tiltAtHalf(2048, 1, 0.45);
    expect(tilt.left).toBeGreaterThanOrEqual(20);
    expect(tilt.top).toBeGreaterThanOrEqual(20);
  });

  test('height maps are told apart from photos', () => {
    expect(heightMapScore(hemispheres(512, 1, 0.45).image)).toBeGreaterThan(
      0.9
    );
    // Fine detail everywhere, as in a photo, in color and in gray.
    let seed = 7;
    const noise = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const size = 256;
    const gray = new Float32Array(size * size).map(() => noise());
    const red = gray.map((v) => Math.min(1, v * 1.2));
    expect(
      heightMapScore({ width: size, height: size, r: gray, g: gray, b: gray })
    ).toBe(0);
    expect(
      heightMapScore({ width: size, height: size, r: red, g: gray, b: gray })
    ).toBe(0);
  });
});

describe('normal map engine, base color and curvature', () => {
  /** Fine random texture under light that fades from left to right. */
  function litTexture(size: number) {
    let seed = 11;
    const noise = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const plane = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const light = 0.3 + 0.7 * (x / (size - 1));
        plane[y * size + x] = (0.45 + 0.15 * noise()) * light;
      }
    }
    return { width: size, height: size, r: plane, g: plane, b: plane };
  }

  const sideMeans = (plane: Float32Array, size: number) => {
    let left = 0;
    let right = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size / 4; x++) {
        left += plane[y * size + x];
        right += plane[y * size + size - 1 - x];
      }
    }
    return left / right;
  };

  test('the default base color is the image itself', () => {
    const image = litTexture(64);
    const maps = computeMaps(image, { ...DEFAULT_SETTINGS, source: 'image' });
    expect(maps.albedo.r).toBe(image.r);
  });

  test('removing shadows evens out light across the image', () => {
    const size = 256;
    const image = litTexture(size);
    const before = sideMeans(image.r, size);
    const maps = computeMaps(image, {
      ...DEFAULT_SETTINGS,
      source: 'image',
      delight: 1,
    });
    const after = sideMeans(maps.albedo.r, size);
    expect(before).toBeLessThan(0.5);
    expect(after).toBeGreaterThan(0.85);
    // The texture itself survives.
    let spread = 0;
    for (let i = 0; i < 64; i++) {
      spread = Math.max(
        spread,
        Math.abs(maps.albedo.r[i * size + size / 2] - maps.albedo.r[size / 2])
      );
    }
    expect(spread).toBeGreaterThan(0.05);
  });

  test('brightness and contrast change the base color', () => {
    const image = litTexture(64);
    const maps = computeMaps(image, {
      ...DEFAULT_SETTINGS,
      source: 'image',
      brightness: 0.1,
    });
    expect(maps.albedo.r[100]).toBeCloseTo(image.r[100] + 0.1, 5);
  });

  test('curvature: the dome top is convex, its foot concave', () => {
    const maps = imageMode();
    const top = maps.curvature[C * SIZE + C];
    // Where the dome meets the floor, a pixel or two around the rim.
    const foot = Math.min(
      ...[-1, 0, 1, 2].map((d) => maps.curvature[C * SIZE + C + RADIUS + d])
    );
    const corner = maps.curvature[4 * SIZE + 4];
    expect(top).toBeGreaterThan(0.5);
    expect(foot).toBeLessThan(0.5);
    expect(Math.abs(corner - 0.5)).toBeLessThan(0.05);
  });
});
