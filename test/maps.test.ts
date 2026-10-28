import { describe, expect, test } from 'vitest';
import type { RgbImage } from '../src/image';
import {
  computeMaps,
  DEFAULT_SETTINGS,
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
