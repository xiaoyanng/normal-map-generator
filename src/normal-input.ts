import { blur, fitSize, resizePlane } from './image';
import type { NormalFormat } from './maps';

/** A normal map as three 0..1 planes (R = X, G = Y, B = Z). */
export type NormalPlanes = [Float32Array, Float32Array, Float32Array];

export interface FormatGuess {
  format: NormalFormat;
  /**
   * How clearly the map reads as that format, 0 (a coin toss) to 1. Below
   * about 0.2 the map is too flat or too noisy to tell.
   */
  confidence: number;
}

/**
 * A tangent-space normal never points into the surface, so its blue
 * channel stays at mid gray or above. Two-channel maps (BC5 / RG exports)
 * leave blue at 0 or another constant; when many pixels sit below mid
 * gray, Z is rebuilt from X and Y instead. Returns the planes to use.
 */
export function completeNormal(planes: NormalPlanes): NormalPlanes {
  const [r, g, b] = planes;
  const n = b.length;
  const stride = Math.max(1, Math.floor(n / 65536));
  let low = 0;
  let count = 0;
  for (let i = 0; i < n; i += stride, count++) if (b[i] < 0.45) low++;
  if (low / count < 0.05) return planes;
  const z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = r[i] * 2 - 1;
    const y = g[i] * 2 - 1;
    z[i] = Math.sqrt(Math.max(0, 1 - x * x - y * y)) * 0.5 + 0.5;
  }
  return [r, g, z];
}

/**
 * Which green-channel convention a normal map uses. A normal map comes from
 * a real surface, so its two slopes must agree with each other (the slope
 * field has no curl). Read with the wrong green direction, one slope is
 * flipped and they stop agreeing. The convention with the smaller curl
 * wins; measured at 512 px at most, lightly smoothed against noise.
 */
export function detectNormalFormat(
  planes: NormalPlanes,
  width: number,
  height: number
): FormatGuess {
  const fit = fitSize(width, height, 512);
  const small = completeNormal(planes).map((plane) =>
    blur(
      resizePlane(plane, width, height, fit.width, fit.height),
      fit.width,
      fit.height,
      1,
      false
    )
  );
  const w = fit.width;
  const h = fit.height;
  const n = w * h;
  // Slopes with y down, read as OpenGL (green up).
  const p = new Float32Array(n);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const nx = small[0][i] * 2 - 1;
    const ny = small[1][i] * 2 - 1;
    const nz = Math.max(0.1, small[2][i] * 2 - 1);
    p[i] = -nx / nz;
    q[i] = ny / nz;
  }
  // Curl of (p, q) and of (p, -q), the DirectX reading.
  let opengl = 0;
  let directx = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const dpdy = (p[i + w] - p[i - w]) / 2;
      const dqdx = (q[i + 1] - q[i - 1]) / 2;
      opengl += (dpdy - dqdx) ** 2;
      directx += (dpdy + dqdx) ** 2;
    }
  }
  const total = opengl + directx;
  if (total === 0) return { format: 'opengl', confidence: 0 };
  return opengl <= directx
    ? { format: 'opengl', confidence: (directx - opengl) / total }
    : { format: 'directx', confidence: (opengl - directx) / total };
}
