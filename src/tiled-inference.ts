/**
 * Runs a fixed-size image-to-normals model (256 px tiles, one grayscale
 * channel in, three normal channels out) over an image of any size.
 *
 * The image is padded on every side, covered by evenly spaced tiles that
 * overlap by at least `overlap` pixels, and each tile's prediction is added
 * with a flat-topped window weight (full weight in the middle, easing to
 * zero over the overlap). Dividing by the summed weights gives a smooth
 * cross-fade wherever tiles meet, and the image's own edges sit in the
 * full-weight part of a tile, away from the model's weaker tile borders.
 */

export const TILE = 256;

/** Runs the model on one TILE x TILE tile; returns 3 * TILE * TILE values. */
export type TileRunner = (tile: Float32Array) => Promise<Float32Array>;

export interface TileOptions {
  /** Least overlap between neighbouring tiles, in pixels. */
  overlap?: number;
  /** Pad by wrapping around (tileable images) instead of mirroring. */
  wrap?: boolean;
  onProgress?: (done: number, total: number) => void;
}

/** Source index for a padded position `i` on an axis of length `n`. */
function sourceIndex(i: number, n: number, wrap: boolean): number {
  if (n === 1) return 0;
  if (wrap) return ((i % n) + n) % n;
  // Mirror without repeating the edge pixel: -1 reads 1, n reads n - 2.
  const period = 2 * (n - 1);
  const m = ((i % period) + period) % period;
  return m < n ? m : period - m;
}

/** Tile start positions covering `length`, evenly spread. */
export function tileStarts(length: number, overlap: number): number[] {
  if (length <= TILE) return [0];
  const count = Math.ceil((length - TILE) / (TILE - overlap)) + 1;
  const step = (length - TILE) / (count - 1);
  return Array.from({ length: count }, (_, i) => Math.round(i * step));
}

/** Per-axis tile weight: eases in and out over `ramp` pixels. */
function windowWeights(ramp: number): Float32Array {
  const weights = new Float32Array(TILE).fill(1);
  for (let i = 0; i < ramp; i++) {
    const w = Math.sin((Math.PI / 2) * ((i + 0.5) / ramp)) ** 2;
    weights[i] = w;
    weights[TILE - 1 - i] = w;
  }
  return weights;
}

/**
 * Normal map (three planes, 0..1 encoded) for a grayscale image in 0..1.
 */
export async function inferNormals(
  gray: Float32Array,
  width: number,
  height: number,
  run: TileRunner,
  options: TileOptions = {}
): Promise<[Float32Array, Float32Array, Float32Array]> {
  const overlap = options.overlap ?? 64;
  const wrap = options.wrap ?? false;
  // Pad by the overlap, and up to one tile for small images.
  const padX = Math.max(overlap, Math.ceil((TILE - width) / 2));
  const padY = Math.max(overlap, Math.ceil((TILE - height) / 2));
  const pw = width + 2 * padX;
  const ph = height + 2 * padY;
  const padded = new Float32Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    const row = sourceIndex(y - padY, height, wrap) * width;
    for (let x = 0; x < pw; x++) {
      padded[y * pw + x] = gray[row + sourceIndex(x - padX, width, wrap)];
    }
  }

  const xs = tileStarts(pw, overlap);
  const ys = tileStarts(ph, overlap);
  const total = xs.length * ys.length;
  const axis = windowWeights(overlap);
  const sums = [
    new Float32Array(pw * ph),
    new Float32Array(pw * ph),
    new Float32Array(pw * ph),
  ];
  const weightSum = new Float32Array(pw * ph);
  const tile = new Float32Array(TILE * TILE);
  const area = TILE * TILE;
  let done = 0;
  options.onProgress?.(0, total);
  for (const top of ys) {
    for (const left of xs) {
      for (let y = 0; y < TILE; y++) {
        const from = (top + y) * pw + left;
        tile.set(padded.subarray(from, from + TILE), y * TILE);
      }
      const prediction = await run(tile);
      for (let y = 0; y < TILE; y++) {
        const wy = axis[y];
        const rowStart = (top + y) * pw + left;
        for (let x = 0; x < TILE; x++) {
          const w = wy * axis[x];
          const at = rowStart + x;
          const k = y * TILE + x;
          sums[0][at] += prediction[k] * w;
          sums[1][at] += prediction[area + k] * w;
          sums[2][at] += prediction[2 * area + k] * w;
          weightSum[at] += w;
        }
      }
      options.onProgress?.(++done, total);
    }
  }

  const n = width * height;
  const out: [Float32Array, Float32Array, Float32Array] = [
    new Float32Array(n),
    new Float32Array(n),
    new Float32Array(n),
  ];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y + padY) * pw + x + padX;
      const w = weightSum[at] || 1;
      const nx = sums[0][at] / w - 0.5;
      const ny = sums[1][at] / w - 0.5;
      const nz = sums[2][at] / w - 0.5;
      const len = Math.hypot(nx, ny, nz) || 1;
      const i = y * width + x;
      out[0][i] = (nx / len) * 0.5 + 0.5;
      out[1][i] = (ny / len) * 0.5 + 0.5;
      out[2][i] = (nz / len) * 0.5 + 0.5;
    }
  }
  return out;
}
