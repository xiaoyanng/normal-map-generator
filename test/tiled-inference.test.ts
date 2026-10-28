import { describe, expect, test } from 'vitest';
import {
  inferNormals,
  TILE,
  type TileRunner,
  tileStarts,
} from '../src/tiled-inference';

/**
 * A stand-in model that maps each pixel on its own (no neighbours), so
 * every tile agrees and the stitched result must equal a direct per-pixel
 * answer, wherever the tiles are cut.
 */
const pixelModel: TileRunner = async (tile) => {
  const area = TILE * TILE;
  const out = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    out[i] = tile[i];
    out[area + i] = 1 - tile[i];
    out[2 * area + i] = 1;
  }
  return out;
};

function expected(v: number): [number, number, number] {
  const n = [v - 0.5, 0.5 - v, 0.5];
  const len = Math.hypot(...n);
  return n.map((c) => (c / len) * 0.5 + 0.5) as [number, number, number];
}

function gradient(width: number, height: number): Float32Array {
  const gray = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      gray[y * width + x] = ((x * 7 + y * 13) % 97) / 96;
    }
  }
  return gray;
}

describe('tiled inference', () => {
  test.each([
    [100, 60],
    [256, 256],
    [300, 257],
    [1024, 700],
  ])('stitches a %ix%i image without moving a pixel', async (w, h) => {
    const gray = gradient(w, h);
    for (const wrap of [false, true]) {
      const normal = await inferNormals(gray, w, h, pixelModel, { wrap });
      let worst = 0;
      for (let i = 0; i < w * h; i++) {
        const want = expected(gray[i]);
        for (let c = 0; c < 3; c++) {
          worst = Math.max(worst, Math.abs(normal[c][i] - want[c]));
        }
      }
      expect(worst).toBeLessThan(1e-5);
    }
  });

  test('tiles cover the whole length and overlap by at least the minimum', () => {
    for (const length of [256, 300, 513, 1152, 2048 + 128]) {
      const starts = tileStarts(length, 64);
      expect(starts[0]).toBe(0);
      expect(starts.at(-1)).toBe(Math.max(0, length - TILE));
      for (let i = 1; i < starts.length; i++) {
        expect(TILE - (starts[i] - starts[i - 1])).toBeGreaterThanOrEqual(64);
      }
    }
  });

  test('reports progress once per tile', async () => {
    const seen: number[] = [];
    await inferNormals(gradient(600, 300), 600, 300, pixelModel, {
      onProgress: (done, total) => seen.push(done / total),
    });
    expect(seen[0]).toBe(0);
    expect(seen.at(-1)).toBe(1);
  });
});
