/**
 * Float image helpers for the map engine. Pure functions on typed arrays, so
 * the same code runs in the browser worker and in Node (sample scripts).
 * Planes are row-major `Float32Array`s of `width * height`.
 */

export interface Plane {
  width: number;
  height: number;
  data: Float32Array;
}

/** RGB as three planes in 0..1. */
export interface RgbImage {
  width: number;
  height: number;
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
}

export function rgbFromRgba(
  rgba: ArrayLike<number>,
  width: number,
  height: number
): RgbImage {
  const n = width * height;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = rgba[i * 4] / 255;
    g[i] = rgba[i * 4 + 1] / 255;
    b[i] = rgba[i * 4 + 2] / 255;
  }
  return { width, height, r, g, b };
}

/** Rec. 709 luminance. */
export function luminance(image: RgbImage): Float32Array {
  const n = image.width * image.height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = 0.2126 * image.r[i] + 0.7152 * image.g[i] + 0.0722 * image.b[i];
  }
  return out;
}

/** Plain mean of the channels, what DeepBump was trained on. */
export function channelMean(image: RgbImage): Float32Array {
  const n = image.width * image.height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (image.r[i] + image.g[i] + image.b[i]) / 3;
  }
  return out;
}

function wrapIndex(i: number, n: number): number {
  const m = i % n;
  return m < 0 ? m + n : m;
}

function clampIndex(i: number, n: number): number {
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

/** One box blur pass along rows (`horizontal`) or columns. */
function boxPass(
  src: Float32Array,
  dst: Float32Array,
  width: number,
  height: number,
  radius: number,
  horizontal: boolean,
  wrap: boolean
) {
  const len = horizontal ? width : height;
  const lines = horizontal ? height : width;
  const step = horizontal ? 1 : width;
  const lineStep = horizontal ? width : 1;
  const index = wrap ? wrapIndex : clampIndex;
  const size = radius * 2 + 1;
  for (let line = 0; line < lines; line++) {
    const base = line * lineStep;
    let sum = 0;
    for (let k = -radius; k <= radius; k++) {
      sum += src[base + index(k, len) * step];
    }
    for (let i = 0; i < len; i++) {
      dst[base + i * step] = sum / size;
      sum +=
        src[base + index(i + radius + 1, len) * step] -
        src[base + index(i - radius, len) * step];
    }
  }
}

/**
 * Gaussian-like blur (three box passes each way). `sigma` in pixels; edges
 * wrap for tileable images, else clamp.
 */
export function blur(
  data: Float32Array,
  width: number,
  height: number,
  sigma: number,
  wrap: boolean
): Float32Array {
  // Box radius whose three passes match the sigma.
  const radius = Math.max(
    0,
    Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2)
  );
  const out = Float32Array.from(data);
  if (radius === 0) return out;
  const tmp = new Float32Array(data.length);
  const r = Math.min(
    radius,
    Math.max(1, Math.floor(Math.min(width, height) / 2) - 1)
  );
  for (let pass = 0; pass < 3; pass++) {
    boxPass(out, tmp, width, height, r, true, wrap);
    boxPass(tmp, out, width, height, r, false, wrap);
  }
  return out;
}

/** Bilinear resize of one plane. */
export function resizePlane(
  data: Float32Array,
  width: number,
  height: number,
  toWidth: number,
  toHeight: number
): Float32Array {
  if (width === toWidth && height === toHeight) return Float32Array.from(data);
  const out = new Float32Array(toWidth * toHeight);
  const sx = width / toWidth;
  const sy = height / toHeight;
  // Shrinking a lot: average first so small details do not alias.
  const src =
    sx > 2 || sy > 2
      ? blur(data, width, height, Math.max(sx, sy) / 3, false)
      : data;
  for (let y = 0; y < toHeight; y++) {
    const fy = Math.min(height - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(height - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < toWidth; x++) {
      const fx = Math.min(width - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(width - 1, x0 + 1);
      const tx = fx - x0;
      const a = src[y0 * width + x0] * (1 - tx) + src[y0 * width + x1] * tx;
      const b = src[y1 * width + x0] * (1 - tx) + src[y1 * width + x1] * tx;
      out[y * toWidth + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

export function resizeRgb(
  image: RgbImage,
  toWidth: number,
  toHeight: number
): RgbImage {
  const { width, height } = image;
  return {
    width: toWidth,
    height: toHeight,
    r: resizePlane(image.r, width, height, toWidth, toHeight),
    g: resizePlane(image.g, width, height, toWidth, toHeight),
    b: resizePlane(image.b, width, height, toWidth, toHeight),
  };
}

/** Size that fits `max` on the longest side, keeping the aspect ratio. */
export function fitSize(
  width: number,
  height: number,
  max: number
): { width: number; height: number } {
  const scale = Math.min(1, max / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/** Values at the given fractions (0..1) of the sorted data, from a sample. */
export function percentiles(data: Float32Array, fractions: number[]): number[] {
  const stride = Math.max(1, Math.floor(data.length / 65536));
  const sample: number[] = [];
  for (let i = 0; i < data.length; i += stride) sample.push(data[i]);
  sample.sort((a, b) => a - b);
  return fractions.map(
    (f) => sample[Math.min(sample.length - 1, Math.floor(f * sample.length))]
  );
}

/** Rescales so the 0.5% and 99.5% points land on 0 and 1, clamped. */
export function normalizeRange(data: Float32Array): Float32Array {
  const [lo, hi] = percentiles(data, [0.005, 0.995]);
  const span = hi - lo || 1;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - lo) / span;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/**
 * Makes an image tile without visible seams: four copies offset by half the
 * size are mixed with weights that are zero on each copy's own seam, and the
 * mix keeps the original contrast (variance-preserving blend).
 */
export function makeSeamless(image: RgbImage): RgbImage {
  const { width, height } = image;
  const hx = Math.floor(width / 2);
  const hy = Math.floor(height / 2);
  const wx = new Float32Array(width);
  const wy = new Float32Array(height);
  for (let x = 0; x < width; x++) wx[x] = Math.sin((Math.PI * x) / width) ** 2;
  for (let y = 0; y < height; y++) {
    wy[y] = Math.sin((Math.PI * y) / height) ** 2;
  }
  const channel = (src: Float32Array) => {
    let mean = 0;
    for (let i = 0; i < src.length; i++) mean += src[i];
    mean /= src.length;
    const out = new Float32Array(src.length);
    for (let y = 0; y < height; y++) {
      const y1 = (y + hy) % height;
      const a0 = wy[y];
      const a1 = 1 - a0;
      for (let x = 0; x < width; x++) {
        const x1 = (x + hx) % width;
        const b0 = wx[x];
        const b1 = 1 - b0;
        const w00 = a0 * b0;
        const w01 = a0 * b1;
        const w10 = a1 * b0;
        const w11 = a1 * b1;
        const v =
          w00 * (src[y * width + x] - mean) +
          w01 * (src[y * width + x1] - mean) +
          w10 * (src[y1 * width + x] - mean) +
          w11 * (src[y1 * width + x1] - mean);
        const norm = Math.sqrt(w00 * w00 + w01 * w01 + w10 * w10 + w11 * w11);
        const out1 = mean + v / norm;
        out[y * width + x] = out1 < 0 ? 0 : out1 > 1 ? 1 : out1;
      }
    }
    return out;
  };
  return {
    width,
    height,
    r: channel(image.r),
    g: channel(image.g),
    b: channel(image.b),
  };
}
