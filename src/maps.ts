import {
  blur,
  luminance,
  normalizeRange,
  percentiles,
  type RgbImage,
} from './image';
import { heightFromSlopes } from './integrate';

/** Where the surface shape comes from. */
export type SurfaceSource = 'ai' | 'image';

/** Green channel convention: OpenGL (Y+, Blender, Unity, Godot) or DirectX. */
export type NormalFormat = 'opengl' | 'directx';

export interface MapSettings {
  source: SurfaceSource;
  /** Bump strength, 1 = default. */
  strength: number;
  /** Weights of the large, medium and fine bumps, 1 = default. */
  large: number;
  medium: number;
  fine: number;
  /** Treat bright as low instead of high (image source only). */
  invert: boolean;
  /**
   * Image source only: drop the largest shapes, for photos lit unevenly
   * (one side brighter) where they come from the light, not the surface.
   * Off by default, so a height map's domes and slopes stay.
   */
  flattenLighting: boolean;
  format: NormalFormat;
  /** The source tiles (made seamless), so filters wrap at the edges. */
  tile: boolean;
  /** Roughness: contrast and brightness offset, invert for glossy darks. */
  roughnessContrast: number;
  roughnessLevel: number;
  roughnessInvert: boolean;
  /** Ambient occlusion strength, 1 = default. */
  aoStrength: number;
}

export const DEFAULT_SETTINGS: MapSettings = {
  source: 'ai',
  strength: 1,
  large: 1,
  medium: 1,
  fine: 1,
  invert: false,
  flattenLighting: false,
  format: 'opengl',
  tile: false,
  roughnessContrast: 1,
  roughnessLevel: 0,
  roughnessInvert: false,
  aoStrength: 1,
};

/** All maps at the working size. Values are 0..1 (normal channels encoded). */
export interface MapSet {
  width: number;
  height: number;
  normal: [Float32Array, Float32Array, Float32Array];
  /** Height and displacement (same data, exported at 16 bits). */
  heightMap: Float32Array;
  roughness: Float32Array;
  ao: Float32Array;
}

/** Slopes (dh/dx, dh/dy with y down) of a height plane, Sobel per pixel. */
function sobel(
  h: Float32Array,
  width: number,
  height: number,
  wrap: boolean
): [Float32Array, Float32Array] {
  const p = new Float32Array(width * height);
  const q = new Float32Array(width * height);
  const ix = (x: number) =>
    wrap ? (x + width) % width : x < 0 ? 0 : x >= width ? width - 1 : x;
  const iy = (y: number) =>
    wrap ? (y + height) % height : y < 0 ? 0 : y >= height ? height - 1 : y;
  for (let y = 0; y < height; y++) {
    const ym = iy(y - 1) * width;
    const y0 = y * width;
    const yp = iy(y + 1) * width;
    for (let x = 0; x < width; x++) {
      const xm = ix(x - 1);
      const xp = ix(x + 1);
      const tl = h[ym + xm];
      const t = h[ym + x];
      const tr = h[ym + xp];
      const l = h[y0 + xm];
      const r = h[y0 + xp];
      const bl = h[yp + xm];
      const b = h[yp + x];
      const br = h[yp + xp];
      p[y0 + x] = (tr + 2 * r + br - tl - 2 * l - bl) / 8;
      q[y0 + x] = (bl + 2 * b + br - tl - 2 * t - tr) / 8;
    }
  }
  return [p, q];
}

/** Blur sizes of the detail bands, in pixels at this working size. */
function bandSigmas(width: number, height: number) {
  const k = Math.max(width, height) / 1024;
  return { fine: 1.5 * k, medium: 6 * k, large: 28 * k };
}

/**
 * Reweights detail bands of a plane: fine (below the fine blur), medium,
 * large, and the base below the large blur.
 */
function reweight(
  data: Float32Array,
  width: number,
  height: number,
  wrap: boolean,
  w: { fine: number; medium: number; large: number; base: number }
): Float32Array {
  const s = bandSigmas(width, height);
  const b1 = blur(data, width, height, s.fine, wrap);
  const b2 = blur(data, width, height, s.medium, wrap);
  const b3 = blur(data, width, height, s.large, wrap);
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] =
      w.fine * (data[i] - b1[i]) +
      w.medium * (b1[i] - b2[i]) +
      w.large * (b2[i] - b3[i]) +
      w.base * b3[i];
  }
  return out;
}

function rms(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i] + b[i] * b[i];
  return Math.sqrt(sum / a.length) || 1;
}

/** Slope a default image-mode map aims for (about 17 degrees). */
const TARGET_SLOPE = 0.3;

/**
 * A clean height map (grayscale, smooth) is drawn with its real slopes:
 * white stands this fraction of the image size above black. Photos keep the
 * target-slope scaling, which suits their noisy brightness.
 */
const HEIGHT_MAP_DEPTH = 0.2;

/**
 * How much the image looks like a clean height map, 0 (a photo) to 1.
 * Grayscale, and little of its slope comes from the finest detail: photos
 * measure 0.86 or more, a smooth dome about 0.3, a dome with light noise
 * about 0.65. The blend between 0.5 and 0.75 avoids a sudden jump.
 */
function heightMapShare(image: RgbImage, lum: Float32Array): number {
  const { width, height } = image;
  const n = width * height;
  let chroma = 0;
  for (let i = 0; i < n; i++) {
    chroma +=
      Math.abs(image.r[i] - image.g[i]) + Math.abs(image.g[i] - image.b[i]);
  }
  if (chroma / n > 0.01) return 0;
  const sigma = Math.max(1, (3 * Math.max(width, height)) / 1024);
  const smooth = blur(lum, width, height, sigma, false);
  const fine = new Float32Array(n);
  for (let i = 0; i < n; i++) fine[i] = lum[i] - smooth[i];
  const [p, q] = sobel(lum, width, height, false);
  const [fp, fq] = sobel(fine, width, height, false);
  const share = rms(fp, fq) / rms(p, q);
  return Math.min(1, Math.max(0, (0.75 - share) / 0.25));
}

/**
 * Computes every map from the (already resized, already seamless) source.
 * `aiNormal` is the model's normal map at the same size, OpenGL encoded; it
 * is required when `settings.source` is `ai`.
 */
export function computeMaps(
  image: RgbImage,
  settings: MapSettings,
  aiNormal?: [Float32Array, Float32Array, Float32Array]
): MapSet {
  const { width, height } = image;
  const n = width * height;
  const wrap = settings.tile;
  const lum = luminance(image);
  const weights = {
    fine: settings.fine,
    medium: settings.medium,
    large: settings.large,
  };

  let p: Float32Array;
  let q: Float32Array;
  let heightPlane: Float32Array;

  if (settings.source === 'ai' && aiNormal) {
    // Slopes from the predicted normals (OpenGL: green up, so dh/dy = ny/nz
    // with y down), reweighted band by band. Blurring slopes is the same as
    // blurring the height they come from.
    const rawP = new Float32Array(n);
    const rawQ = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const nx = aiNormal[0][i] * 2 - 1;
      const ny = aiNormal[1][i] * 2 - 1;
      const nz = Math.max(0.05, aiNormal[2][i] * 2 - 1);
      rawP[i] = -nx / nz;
      rawQ[i] = ny / nz;
    }
    const w = { ...weights, base: 1 };
    p = reweight(rawP, width, height, wrap, w);
    q = reweight(rawQ, width, height, wrap, w);
    for (let i = 0; i < n; i++) {
      p[i] *= settings.strength;
      q[i] *= settings.strength;
    }
    heightPlane = heightFromSlopes(p, q, width, height, wrap);
    // Integration drifts at the largest scale; keep shapes up to about a
    // tenth of the image and drop the slow gradient across it.
    const drift = blur(
      heightPlane,
      width,
      height,
      Math.max(width, height) / 20,
      wrap
    );
    for (let i = 0; i < n; i++) heightPlane[i] -= drift[i];
  } else {
    // Brightness as height. The base band holds the largest shapes; it is
    // dropped only on request, to remove a lighting gradient from a photo.
    const base = settings.invert ? lum.map((v) => 1 - v) : lum;
    heightPlane = reweight(base, width, height, wrap, {
      ...weights,
      base: settings.flattenLighting ? 0 : 1,
    });
    [p, q] = sobel(heightPlane, width, height, wrap);
    // Photos: the same default look at any size and contrast, scaled to a
    // target slope. Clean height maps: their real slopes, so a smooth dome
    // keeps a clear direction all the way up its sides (a global target
    // would flatten them against the steep rim). Capped at three times the
    // photo scale in case a photo is taken for a height map.
    const photoScale = TARGET_SLOPE / rms(p, q);
    const clean = heightMapShare(image, lum);
    const depthScale = Math.min(
      3 * photoScale,
      Math.max(photoScale, HEIGHT_MAP_DEPTH * Math.max(width, height))
    );
    const scale =
      (photoScale + (depthScale - photoScale) * clean) * settings.strength;
    for (let i = 0; i < n; i++) {
      p[i] *= scale;
      q[i] *= scale;
    }
  }

  const sign = settings.format === 'opengl' ? 1 : -1;
  const nr = new Float32Array(n);
  const ng = new Float32Array(n);
  const nb = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = -p[i];
    const y = sign * q[i];
    const len = Math.sqrt(x * x + y * y + 1);
    nr[i] = (x / len) * 0.5 + 0.5;
    ng[i] = (y / len) * 0.5 + 0.5;
    nb[i] = (1 / len) * 0.5 + 0.5;
  }

  const heightMap = normalizeRange(heightPlane);
  const ao = aoMap(heightMap, width, height, wrap, settings.aoStrength);
  return {
    width,
    height,
    normal: [nr, ng, nb],
    heightMap,
    roughness: roughnessMap(lum, ao, settings),
    ao,
  };
}

/**
 * Most real surfaces are fairly rough; crevices (low AO) and dusty, chalky
 * bright spots read rougher, darker spots a little smoother. Contrast and
 * level tune it; invert suits glossy materials with dark matte parts.
 */
function roughnessMap(
  lum: Float32Array,
  ao: Float32Array,
  settings: MapSettings
): Float32Array {
  const [mid] = percentiles(lum, [0.5]);
  const out = new Float32Array(lum.length);
  const c = settings.roughnessContrast;
  for (let i = 0; i < lum.length; i++) {
    let v =
      0.66 +
      c * (0.55 * (lum[i] - mid) + 0.3 * (1 - ao[i])) +
      settings.roughnessLevel;
    if (settings.roughnessInvert) v = 1 - v;
    out[i] = v < 0.02 ? 0.02 : v > 1 ? 1 : v;
  }
  return out;
}

/**
 * Cavity-style ambient occlusion: how far a point sits below the average
 * height around it, at three radii.
 */
function aoMap(
  h: Float32Array,
  width: number,
  height: number,
  wrap: boolean,
  strength: number
): Float32Array {
  const k = Math.max(width, height) / 1024;
  const radii = [3 * k, 10 * k, 30 * k];
  const weights = [1.2, 1, 0.8];
  const out = new Float32Array(h.length).fill(0);
  radii.forEach((r, j) => {
    const b = blur(h, width, height, r, wrap);
    for (let i = 0; i < h.length; i++) {
      const d = b[i] - h[i];
      if (d > 0) out[i] += weights[j] * d;
    }
  });
  for (let i = 0; i < h.length; i++) {
    const v = 1 - 1.15 * strength * out[i];
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v ** 1.2;
  }
  return out;
}
