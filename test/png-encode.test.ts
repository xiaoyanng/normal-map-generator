import { deflateSync, inflateSync } from 'node:zlib';
import { describe, expect, test } from 'vitest';
import {
  encodeGray16Png,
  encodeRgbaPng,
} from '../src/encode';

const deflate = async (data: Uint8Array) => deflateSync(data);

/** Header fields and the inflated pixel rows of a single-IDAT PNG. */
function readPng(png: Uint8Array) {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  let offset = 8;
  let header = { width: 0, height: 0, depth: 0, colorType: 0 };
  let pixels = new Uint8Array(0);
  while (offset < png.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...png.subarray(offset + 4, offset + 8));
    const data = png.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      header = {
        width: view.getUint32(offset + 8),
        height: view.getUint32(offset + 12),
        depth: data[8],
        colorType: data[9],
      };
    }
    if (type === 'IDAT') pixels = inflateSync(data);
    offset += 12 + length;
  }
  return { ...header, pixels };
}

describe('PNG encoders', () => {
  test('RGBA keeps the color of fully transparent pixels', async () => {
    const plane = (...values: number[]) => Float32Array.from(values);
    const png = await encodeRgbaPng(
      [plane(1, 0), plane(0.5, 0.25), plane(0, 1), plane(0, 1)],
      2,
      1,
      deflate
    );
    const { width, height, depth, colorType, pixels } = readPng(png);
    expect({ width, height, depth, colorType }).toEqual({
      width: 2,
      height: 1,
      depth: 8,
      colorType: 6,
    });
    // Filter byte, then R G B A per pixel; alpha 0 does not wipe the color.
    expect([...pixels]).toEqual([0, 255, 128, 0, 0, 0, 64, 255, 255]);
  });

  test('16-bit grayscale stores 65,536 levels', async () => {
    const png = await encodeGray16Png(
      Float32Array.from([0, 0.5, 1]),
      3,
      1,
      deflate
    );
    const { depth, colorType, pixels } = readPng(png);
    expect({ depth, colorType }).toEqual({ depth: 16, colorType: 0 });
    expect([...pixels]).toEqual([0, 0, 0, 128, 0, 255, 255]);
  });
});
