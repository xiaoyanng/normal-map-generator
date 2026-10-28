import type { MapSet } from './maps';
import type { RgbImage } from './image';

/**
 * Lights a flat surface with the maps, in software: the same model as the
 * live WebGL preview (diffuse + GGX-like highlight, ambient occlusion), used
 * for still images such as the example cards. `light` points from the
 * surface towards the light, in OpenGL tangent space (x right, y up, z out).
 * Returns RGBA bytes.
 */
export function shade(
  albedo: RgbImage,
  maps: MapSet,
  light: [number, number, number],
  options: { format: 'opengl' | 'directx'; exposure?: number } = {
    format: 'opengl',
  }
): Uint8ClampedArray {
  const { width, height } = maps;
  const n = width * height;
  const out = new Uint8ClampedArray(n * 4);
  const ll = Math.hypot(...light);
  const lx = light[0] / ll;
  const ly = light[1] / ll;
  const lz = light[2] / ll;
  // Half vector with the viewer straight above.
  const hl = Math.hypot(lx, ly, lz + 1);
  const hx = lx / hl;
  const hy = ly / hl;
  const hz = (lz + 1) / hl;
  const sign = options.format === 'opengl' ? 1 : -1;
  const exposure = options.exposure ?? 1;
  const toSrgb = (v: number) => 255 * Math.min(1, Math.max(0, v) ** (1 / 2.2));
  const toLinear = (v: number) => v ** 2.2;
  for (let i = 0; i < n; i++) {
    const nx = maps.normal[0][i] * 2 - 1;
    const ny = sign * (maps.normal[1][i] * 2 - 1);
    const nz = maps.normal[2][i] * 2 - 1;
    const ndl = Math.max(0, nx * lx + ny * ly + nz * lz);
    const ndh = Math.max(0, nx * hx + ny * hy + nz * hz);
    const rough = maps.roughness[i];
    const a = Math.max(0.04, rough * rough);
    const d = ndh * ndh * (a * a - 1) + 1;
    const spec = ((a * a) / (Math.PI * d * d)) * 0.04 * ndl;
    const ao = maps.ao[i];
    const ambient = 0.18 * ao;
    const lit = (c: number) =>
      toSrgb(
        exposure *
          (toLinear(c) * (ambient + 1.1 * ndl * (0.6 + 0.4 * ao)) + spec)
      );
    out[i * 4] = lit(albedo.r[i]);
    out[i * 4 + 1] = lit(albedo.g[i]);
    out[i * 4 + 2] = lit(albedo.b[i]);
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Interleaves planes (0..1) into RGBA bytes for display or 8-bit PNG. */
export function toRgba(
  r: Float32Array,
  g: Float32Array = r,
  b: Float32Array = r
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(r.length * 4);
  for (let i = 0; i < r.length; i++) {
    out[i * 4] = r[i] * 255 + 0.5;
    out[i * 4 + 1] = g[i] * 255 + 0.5;
    out[i * 4 + 2] = b[i] * 255 + 0.5;
    out[i * 4 + 3] = 255;
  }
  return out;
}

function sample(
  plane: Float32Array,
  width: number,
  height: number,
  u: number,
  v: number
): number {
  const x = (((u % 1) + 1) % 1) * width - 0.5;
  const y = Math.min(height - 1, Math.max(0, v * height - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const xa = ((x0 % width) + width) % width;
  const xb = (xa + 1) % width;
  const ya = y0;
  const yb = Math.min(height - 1, y0 + 1);
  const a = plane[ya * width + xa] * (1 - tx) + plane[ya * width + xb] * tx;
  const b = plane[yb * width + xa] * (1 - tx) + plane[yb * width + xb] * tx;
  return a * (1 - ty) + b * ty;
}

/**
 * A material ball: the maps on a sphere, lit from the upper left, seen
 * straight on. Same mapping as the live preview's sphere (the texture goes
 * twice around the equator). Transparent outside the ball; RGBA bytes.
 */
export function shadeSphere(
  albedo: RgbImage,
  maps: MapSet,
  size: number,
  light: [number, number, number] = [-0.55, 0.6, 0.58]
): Uint8ClampedArray {
  const { width, height } = maps;
  const out = new Uint8ClampedArray(size * size * 4);
  const ll = Math.hypot(...light);
  const [lx, ly, lz] = light.map((v) => v / ll);
  const hl = Math.hypot(lx, ly, lz + 1);
  const [hx, hy, hz] = [lx / hl, ly / hl, (lz + 1) / hl];
  const radius = 0.94;
  const toSrgb = (v: number) => 255 * Math.min(1, Math.max(0, v) ** (1 / 2.2));
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const x = ((px + 0.5) / size) * 2 - 1;
      const y = 1 - ((py + 0.5) / size) * 2;
      const r2 = (x * x + y * y) / (radius * radius);
      if (r2 >= 1) continue;
      const sx = x / radius;
      const sy = y / radius;
      const sz = Math.sqrt(1 - sx * sx - sy * sy);
      const theta = Math.acos(sy);
      const phi = Math.atan2(-sz, sx);
      const u = phi / Math.PI;
      const v = theta / Math.PI;
      const tx = -Math.sin(phi);
      const tz = -Math.cos(phi);
      // B = N x T
      const bx = sy * tz;
      const by = sz * tx - sx * tz;
      const bz = -sy * tx;
      const mx = sample(maps.normal[0], width, height, u, v) * 2 - 1;
      const my = sample(maps.normal[1], width, height, u, v) * 2 - 1;
      const mz = sample(maps.normal[2], width, height, u, v) * 2 - 1;
      let nx = tx * mx + bx * my + sx * mz;
      let ny = by * my + sy * mz;
      let nz = tz * mx + bz * my + sz * mz;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl;
      ny /= nl;
      nz /= nl;
      const ndl = Math.max(0, nx * lx + ny * ly + nz * lz);
      const ndh = Math.max(0, nx * hx + ny * hy + nz * hz);
      const rough = sample(maps.roughness, width, height, u, v);
      const ao = sample(maps.ao, width, height, u, v);
      const a = Math.max(0.04, rough * rough);
      const d = ndh * ndh * (a * a - 1) + 1;
      const spec = ((a * a) / (Math.PI * d * d)) * 0.04 * ndl;
      // A soft fill from below-right so the dark side is not black.
      const fill = Math.max(0, 0.4 * nx - 0.3 * ny + 0.5 * nz) * 0.12;
      const shadeOf = (c: number) =>
        toSrgb(
          c ** 2.2 * (0.1 * ao + fill + 1.25 * ndl * (0.55 + 0.45 * ao)) + spec
        );
      const i = (py * size + px) * 4;
      out[i] = shadeOf(sample(albedo.r, width, height, u, v));
      out[i + 1] = shadeOf(sample(albedo.g, width, height, u, v));
      out[i + 2] = shadeOf(sample(albedo.b, width, height, u, v));
      // Soft edge over the last pixel or so.
      const edge = (1 - Math.sqrt(r2)) * size * radius * 0.5;
      out[i + 3] = 255 * Math.min(1, edge);
    }
  }
  return out;
}
