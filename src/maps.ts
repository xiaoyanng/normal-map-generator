import {
  blur,
  fitSize,
  luminance,
  normalizeRange,
  percentiles,
  type RgbImage,
  resizePlane,
} from './image';
import { heightFromSlopes } from './integrate';
import {
  completeNormal,
  detectNormalFormat,
  type FormatGuess,
} from './normal-input';

/**
 * Where the surface shape comes from: the AI model, the image's
 * brightness, or the image itself when it already is a normal map.
 */
export type SurfaceSource = 'ai' | 'image' | 'normal';

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
  /**
   * Normal source only: the green convention of the uploaded normal map,
   * or `auto` to tell it from the map (see `detectNormalFormat`).
   */
  normalInput: NormalFormat | 'auto';
  /** Base color: brightness offset (0 = as is) and contrast (1 = as is). */
  brightness: number;
  contrast: number;
  /**
   * Base color: how much of the light baked into the photo to divide out,
   * 0 (none) to 1: uneven light across it, and with AI normals also the
   * shading of every bump and the darkness of its crevices.
   */
  delight: number;
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
  normalInput: 'auto',
  brightness: 0,
  contrast: 1,
  delight: 0,
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
  /** Convex edges light, cavities dark, flat 0.5. */
  curvature: Float32Array;
  /** Base color: the source after brightness, contrast and delighting. */
  albedo: RgbImage;
  /** Normal source only: the convention the uploaded map was read as. */
  normalInput?: FormatGuess & { detected: boolean };
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
function heightMapShare(image: RgbImage, full: Float32Array): number {
  const n0 = image.width * image.height;
  let chroma = 0;
  for (let i = 0; i < n0; i++) {
    chroma +=
      Math.abs(image.r[i] - image.g[i]) + Math.abs(image.g[i] - image.b[i]);
  }
  if (chroma / n0 > 0.01) return 0;
  // Measured at 1024 px at most, the size it was tuned at: in a larger
  // 8-bit file the one-level steps up a gentle slope would count as fine
  // detail and make a clean height map look like a photo.
  const { width, height } = fitSize(image.width, image.height, 1024);
  const lum =
    width === image.width
      ? full
      : resizePlane(full, image.width, image.height, width, height);
  const n = width * height;
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
 * How much an image looks like a clean height map rather than a photo, 0
 * to 1 (photos score 0). Above 0, AI photo mode is the wrong tool: the
 * model is trained on photos and flattens smooth height maps, so the
 * generator uses the image's own slopes instead.
 */
export function heightMapScore(image: RgbImage): number {
  return heightMapShare(image, luminance(image));
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
  // A normal map as the input: its colors are directions, not a surface
  // color, so the base color is plain mid gray.
  const fromNormal = settings.source === 'normal';
  let normalInput: MapSet['normalInput'];
  let shapeNormal = aiNormal;
  if (fromNormal) {
    const planes = completeNormal([image.r, image.g, image.b]);
    const guess =
      settings.normalInput === 'auto'
        ? detectNormalFormat(planes, width, height)
        : { format: settings.normalInput, confidence: 1 };
    normalInput = { ...guess, detected: settings.normalInput === 'auto' };
    shapeNormal =
      guess.format === 'directx'
        ? [planes[0], planes[1].map((v) => 1 - v), planes[2]]
        : planes;
  }
  const source = fromNormal
    ? grayImage(width, height)
    : adjust(image, settings.brightness, settings.contrast);
  const lum = luminance(source);
  const weights = {
    fine: settings.fine,
    medium: settings.medium,
    large: settings.large,
  };

  let p: Float32Array;
  let q: Float32Array;
  let heightPlane: Float32Array;

  if ((settings.source === 'ai' || fromNormal) && shapeNormal) {
    // Slopes from the predicted (or uploaded) normals (OpenGL: green up,
    // so dh/dy = ny/nz with y down), reweighted band by band. Blurring
    // slopes is the same as blurring the height they come from.
    const rawP = new Float32Array(n);
    const rawQ = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const nx = shapeNormal[0][i] * 2 - 1;
      const ny = shapeNormal[1][i] * 2 - 1;
      const nz = Math.max(0.05, shapeNormal[2][i] * 2 - 1);
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
    // Predicted normals drift when integrated at the largest scale; keep
    // shapes up to about a tenth of the image and drop the slow gradient
    // across it. An uploaded normal map comes from a real height or bake,
    // so its large shapes stay; only a lean across the whole image (from
    // slopes that do not quite agree) is taken out, as a best-fit plane.
    if (fromNormal) {
      removePlane(heightPlane, width, height);
    } else {
      const drift = blur(
        heightPlane,
        width,
        height,
        Math.max(width, height) / 20,
        wrap
      );
      for (let i = 0; i < n; i++) heightPlane[i] -= drift[i];
    }
  } else {
    // Brightness as height. The base band holds the largest shapes; it is
    // dropped only on request, to remove a lighting gradient from a photo.
    const base = settings.invert ? lum.map((v) => 1 - v) : lum;
    heightPlane = reweight(base, width, height, wrap, {
      ...weights,
      base: settings.flattenLighting ? 0 : 1,
    });
    const clean = heightMapShare(image, lum);
    // A clean height map stored in 8 bits climbs a gentle slope one gray
    // level every few pixels; a 3x3 slope reads the flat runs between the
    // steps as level ground. A light blur (more for larger images) turns
    // the steps back into the slope they came from.
    let slopePlane = heightPlane;
    if (clean > 0) {
      const soft = blur(
        heightPlane,
        width,
        height,
        Math.max(width, height) / 1024,
        wrap
      );
      slopePlane = heightPlane.map((v, i) => v + (soft[i] - v) * clean);
    }
    [p, q] = sobel(slopePlane, width, height, wrap);
    // Photos: the same default look at any size and contrast, scaled to a
    // target slope. Clean height maps: their real slopes, so a smooth dome
    // keeps a clear direction all the way up its sides (a global target
    // would flatten them against the steep rim). In between, capped at
    // three times the photo scale in case a photo is taken for a height
    // map; the cap fades out as the image gets clearly clean, because a
    // steep rim makes the photo scale tiny and the cap would flatten it.
    const photoScale = TARGET_SLOPE / rms(p, q);
    const realScale = Math.max(
      photoScale,
      HEIGHT_MAP_DEPTH * Math.max(width, height)
    );
    const capped = Math.min(3 * photoScale, realScale);
    const depthScale = capped + (realScale - capped) * clean;
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
  // Delighting divides by the photo's own shading: from the normals only
  // when they come from the AI (image-mode normals are the brightness
  // itself, and dividing them out would wipe the texture).
  const shapeNormals = settings.source === 'ai' && aiNormal;
  return {
    width,
    height,
    normal: [nr, ng, nb],
    heightMap,
    roughness: roughnessMap(lum, ao, settings),
    ao,
    curvature: curvatureMap(p, q, width, height, wrap),
    albedo: fromNormal
      ? source
      : delightImage(source, lum, settings.delight, wrap, {
          p: shapeNormals ? p : null,
          q: shapeNormals ? q : null,
          ao,
        }),
    normalInput,
  };
}

/** Subtracts the least-squares plane a + b x + c y, in place. */
function removePlane(h: Float32Array, width: number, height: number) {
  // x and y centred, so the three terms are independent.
  let sh = 0;
  let sx = 0;
  let sy = 0;
  let xx = 0;
  let yy = 0;
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = h[y * width + x];
      sh += v;
      sx += v * (x - cx);
      sy += v * (y - cy);
      xx += (x - cx) ** 2;
      yy += (y - cy) ** 2;
    }
  }
  const a = sh / (width * height);
  const b = xx ? sx / xx : 0;
  const c = yy ? sy / yy : 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      h[y * width + x] -= a + b * (x - cx) + c * (y - cy);
    }
  }
}

/** Plain mid gray, the base color when the input is a normal map. */
function grayImage(width: number, height: number): RgbImage {
  const plane = new Float32Array(width * height).fill(0.5);
  return { width, height, r: plane, g: plane, b: plane };
}

/** Brightness offset and contrast around mid gray, per channel. */
function adjust(image: RgbImage, brightness: number, contrast: number) {
  if (brightness === 0 && contrast === 1) return image;
  const f = (plane: Float32Array) =>
    plane.map((v) => {
      const out = (v - 0.5) * contrast + 0.5 + brightness;
      return out < 0 ? 0 : out > 1 ? 1 : out;
    });
  return { ...image, r: f(image.r), g: f(image.g), b: f(image.b) };
}

/**
 * Convexity from the slopes: the negative Laplacian of the height, at a
 * fine and a medium radius, scaled so the strongest edges reach white
 * (and the deepest cavities black) and flat ground stays mid gray.
 */
function curvatureMap(
  p: Float32Array,
  q: Float32Array,
  width: number,
  height: number,
  wrap: boolean
): Float32Array {
  const n = width * height;
  const k = Math.max(width, height) / 1024;
  const ix = (x: number) =>
    wrap ? (x + width) % width : x < 0 ? 0 : x >= width ? width - 1 : x;
  const iy = (y: number) =>
    wrap ? (y + height) % height : y < 0 ? 0 : y >= height ? height - 1 : y;
  const sum = new Float32Array(n);
  for (const sigma of [1 * k, 3 * k]) {
    const bp = blur(p, width, height, sigma, wrap);
    const bq = blur(q, width, height, sigma, wrap);
    const lap = new Float32Array(n);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dp = (bp[y * width + ix(x + 1)] - bp[y * width + ix(x - 1)]) / 2;
        const dq = (bq[iy(y + 1) * width + x] - bq[iy(y - 1) * width + x]) / 2;
        lap[y * width + x] = -(dp + dq);
      }
    }
    const [lo, hi] = percentiles(lap, [0.01, 0.99]);
    const range = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    for (let i = 0; i < n; i++) sum[i] += lap[i] / range;
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = 0.5 + 0.25 * sum[i];
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/**
 * Takes the light baked into a photo out of its colors, by `amount` (0 to
 * 1). The photo is modelled as base color times shading: a slow change of
 * light across the image, plus, when the slopes come from the AI, the
 * light and shade of every bump (fitted from the photo as brightness
 * against surface direction) and the darkening of crevices.
 */
function delightImage(
  image: RgbImage,
  lum: Float32Array,
  amount: number,
  wrap: boolean,
  shape: { p: Float32Array | null; q: Float32Array | null; ao: Float32Array }
): RgbImage {
  if (amount <= 0) return image;
  const { width, height } = image;
  const n = width * height;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += lum[i];
  mean = mean / n || 1;
  // Light at two scales: across the image (a bright corner, a dark side)
  // and across a few stones or boards (the shadow one casts on the next).
  // The finer layer counts at 60% so real color changes survive.
  const size = Math.max(width, height);
  const slow = blur(lum, width, height, size / 8, wrap);
  const mid = blur(lum, width, height, size / 48, wrap);
  const shading = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const across = Math.max(0.03, slow[i] / mean);
    const local = Math.max(0.03, mid[i]) / Math.max(0.03, slow[i]);
    shading[i] = across * local ** 0.6;
  }

  if (shape.p && shape.q) {
    // Brightness (with the slow light divided out) against the surface
    // direction: lum = a + b * nx + c * ny, least squares on a sample.
    const nx = new Float32Array(n);
    const ny = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const len = Math.sqrt(shape.p[i] ** 2 + shape.q[i] ** 2 + 1);
      // OpenGL convention: x right, y up (q is the slope with y down).
      nx[i] = -shape.p[i] / len;
      ny[i] = shape.q[i] / len;
    }
    const m = new Float64Array(9);
    const v = new Float64Array(3);
    const stride = Math.max(1, Math.floor(n / 65536));
    for (let i = 0; i < n; i += stride) {
      const row = [1, nx[i], ny[i]];
      const y = lum[i] / shading[i];
      for (let r = 0; r < 3; r++) {
        v[r] += row[r] * y;
        for (let c = 0; c < 3; c++) m[r * 3 + c] += row[r] * row[c];
      }
    }
    const [a, b, c] = solve3(m, v);
    if (a > 0.02) {
      for (let i = 0; i < n; i++) {
        const bump = 1 + (b * nx[i] + c * ny[i]) / a;
        const cavity = 0.7 + 0.3 * shape.ao[i];
        shading[i] *= Math.max(0.2, bump) * cavity;
      }
    }
  }

  // Divide, then bring the average back to the photo's own brightness.
  const factor = new Float32Array(n);
  let after = 0;
  for (let i = 0; i < n; i++) {
    factor[i] = 1 / Math.min(4, Math.max(0.18, shading[i])) ** amount;
    after += lum[i] * factor[i];
  }
  const keep = mean / (after / n || 1);
  // Shadows lit by the sky are bluer than the sunlit parts; lifted, that
  // tint shows. The more a pixel is lifted, the more its color moves to
  // the photo's average color (its brightness stays).
  const avg = [image.r, image.g, image.b].map((plane) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += plane[i];
    return sum / n / mean;
  });
  const out = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const planes = [image.r, image.g, image.b];
  for (let i = 0; i < n; i++) {
    const gain = factor[i] * keep;
    const neutral = Math.min(1, Math.max(0, (gain - 1.3) / 1.5)) * amount;
    const l = Math.max(1e-4, lum[i]);
    for (let c = 0; c < 3; c++) {
      const tint = planes[c][i] / l;
      const value = l * gain * (tint + (avg[c] - tint) * neutral);
      out[c][i] = value < 0 ? 0 : value > 1 ? 1 : value;
    }
  }
  return { ...image, r: out[0], g: out[1], b: out[2] };
}

/** Solves a 3x3 system (row-major `m`) by Cramer's rule; zeros if singular. */
function solve3(m: Float64Array, v: Float64Array): [number, number, number] {
  const det = (a: ArrayLike<number>) =>
    a[0] * (a[4] * a[8] - a[5] * a[7]) -
    a[1] * (a[3] * a[8] - a[5] * a[6]) +
    a[2] * (a[3] * a[7] - a[4] * a[6]);
  const d = det(m);
  if (Math.abs(d) < 1e-12) return [0, 0, 0];
  const out: number[] = [];
  for (let col = 0; col < 3; col++) {
    const mc = Float64Array.from(m);
    for (let r = 0; r < 3; r++) mc[r * 3 + col] = v[r];
    out.push(det(mc) / d);
  }
  return out as [number, number, number];
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
