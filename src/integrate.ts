import { resizePlane } from './image';

/** In-place radix-2 FFT of one line (`re`, `im` of power-of-two length). */
function fft1d(re: Float64Array, im: Float64Array, inverse: boolean) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const next = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = next;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** 2D FFT in place, rows then columns. */
function fft2d(
  re: Float64Array,
  im: Float64Array,
  width: number,
  height: number,
  inverse: boolean
) {
  const rowRe = new Float64Array(width);
  const rowIm = new Float64Array(width);
  for (let y = 0; y < height; y++) {
    const o = y * width;
    rowRe.set(re.subarray(o, o + width));
    rowIm.set(im.subarray(o, o + width));
    fft1d(rowRe, rowIm, inverse);
    re.set(rowRe, o);
    im.set(rowIm, o);
  }
  const colRe = new Float64Array(height);
  const colIm = new Float64Array(height);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      colRe[y] = re[y * width + x];
      colIm[y] = im[y * width + x];
    }
    fft1d(colRe, colIm, inverse);
    for (let y = 0; y < height; y++) {
      re[y * width + x] = colRe[y];
      im[y * width + x] = colIm[y];
    }
  }
}

function floorPow2(n: number): number {
  return 2 ** Math.floor(Math.log2(Math.max(1, n)));
}

/**
 * Height from a slope field (Frankot-Chellappa). `p` = dh/dx and `q` =
 * dh/dy per pixel, y pointing down the image. The slopes are resampled to a
 * power-of-two grid of at most `maxSize` per side; when the image does not
 * tile they are mirrored first so the FFT sees a periodic field. Returns the
 * height in pixel units at the input size, with zero mean.
 */
export function heightFromSlopes(
  p: Float32Array,
  q: Float32Array,
  width: number,
  height: number,
  wrap: boolean,
  maxSize = 1024
): Float32Array {
  const gw = Math.min(floorPow2(width), maxSize);
  const gh = Math.min(floorPow2(height), maxSize);
  // Slopes per grid cell: one grid cell spans width / gw pixels.
  const sx = width / gw;
  const sy = height / gh;
  const gp = resizePlane(p, width, height, gw, gh);
  const gq = resizePlane(q, width, height, gw, gh);
  const fw = wrap ? gw : gw * 2;
  const fh = wrap ? gh : gh * 2;
  const pr = new Float64Array(fw * fh);
  const qr = new Float64Array(fw * fh);
  for (let y = 0; y < fh; y++) {
    const my = y < gh ? y : fh - 1 - y;
    const fy = y < gh ? 1 : -1;
    for (let x = 0; x < fw; x++) {
      const mx = x < gw ? x : fw - 1 - x;
      const fx = x < gw ? 1 : -1;
      const i = my * gw + mx;
      // Mirroring flips the slope across the mirrored axis.
      pr[y * fw + x] = gp[i] * sx * fx;
      qr[y * fw + x] = gq[i] * sy * fy;
    }
  }
  const pi = new Float64Array(fw * fh);
  const qi = new Float64Array(fw * fh);
  fft2d(pr, pi, fw, fh, false);
  fft2d(qr, qi, fw, fh, false);
  // H = (-i u P - i v Q) / (u^2 + v^2), u and v in radians per cell.
  const hr = new Float64Array(fw * fh);
  const hi = new Float64Array(fw * fh);
  for (let y = 0; y < fh; y++) {
    const v = (2 * Math.PI * (y <= fh / 2 ? y : y - fh)) / fh;
    for (let x = 0; x < fw; x++) {
      const u = (2 * Math.PI * (x <= fw / 2 ? x : x - fw)) / fw;
      const i = y * fw + x;
      const d = u * u + v * v;
      if (d === 0) continue;
      // (-i u)(a + ib) = u b - i u a
      hr[i] = (u * pi[i] + v * qi[i]) / d;
      hi[i] = (-u * pr[i] - v * qr[i]) / d;
    }
  }
  fft2d(hr, hi, fw, fh, true);
  const grid = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) grid[y * gw + x] = hr[y * fw + x];
  }
  return resizePlane(grid, gw, gh, width, height);
}
