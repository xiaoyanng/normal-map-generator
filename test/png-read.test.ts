import { deflateSync, inflateSync } from 'node:zlib';
import sharp from 'sharp';
import { describe, expect, test } from 'vitest';
import {
  flipGreen,
  type Png,
  previewRgba,
  readPng,
  rgbPlanes,
  writePng,
} from '../src/png';

const inflate = async (data: Uint8Array) => inflateSync(data);
const deflate = async (data: Uint8Array) => deflateSync(data);

/** A 64x48 gradient with every channel varying. */
function pixels(channels: 3 | 4) {
  const w = 64;
  const h = 48;
  const out = Buffer.alloc(w * h * channels);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * channels;
      out[i] = (x * 4) & 255;
      out[i + 1] = (y * 5 + x) & 255;
      out[i + 2] = 200 - y;
      // Alpha: fully transparent in the left quarter.
      if (channels === 4) out[i + 3] = x < 16 ? 0 : 128 + y;
    }
  }
  return { w, h, data: out };
}

async function flipped(png: Buffer) {
  const read = (await readPng(new Uint8Array(png), inflate)) as Png;
  expect(read).not.toBeNull();
  flipGreen(read);
  return Buffer.from(await writePng(read, deflate));
}

describe('PNG read and rewrite', () => {
  test.each([
    3, 4,
  ] as const)('%i-channel 8-bit: only green flips, even under alpha 0', async (channels) => {
    const { w, h, data } = pixels(channels);
    const input = await sharp(data, { raw: { width: w, height: h, channels } })
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .toBuffer();
    const output = await flipped(input);
    // Decoded by an independent decoder (sharp, no premultiplying).
    const out = await sharp(output).raw().toBuffer({ resolveWithObject: true });
    expect(out.info.channels).toBe(channels);
    for (let i = 0; i < data.length; i += channels) {
      expect(out.data[i]).toBe(data[i]);
      expect(out.data[i + 1]).toBe(255 - data[i + 1]);
      expect(out.data[i + 2]).toBe(data[i + 2]);
      if (channels === 4) expect(out.data[i + 3]).toBe(data[i + 3]);
    }
  });

  test('a palette image flips its palette, not its indices', async () => {
    const { w, h, data } = pixels(3);
    const input = await sharp(data, {
      raw: { width: w, height: h, channels: 3 },
    })
      .png({ palette: true, colours: 64 })
      .toBuffer();
    const before = await sharp(input).raw().toBuffer();
    const output = await flipped(input);
    const read = (await readPng(new Uint8Array(output), inflate)) as Png;
    expect(read.colorType).toBe(3);
    const after = await sharp(output).raw().toBuffer();
    for (let i = 0; i < before.length; i += 3) {
      expect(after[i + 1]).toBe(255 - before[i + 1]);
      expect(after[i]).toBe(before[i]);
    }
  });

  test('16-bit stays 16-bit with every sample exact', async () => {
    const w = 40;
    const h = 30;
    const pixels16 = new Uint8Array(w * h * 6);
    const sample = (x: number, y: number, c: number) =>
      (x * 1601 + y * 977 + c * 12345) % 65536;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 3; c++) {
          const v = sample(x, y, c);
          const o = (y * w + x) * 6 + c * 2;
          pixels16[o] = v >> 8;
          pixels16[o + 1] = v & 255;
        }
      }
    }
    const source: Png = {
      width: w,
      height: h,
      bitDepth: 16,
      colorType: 2,
      channels: 3,
      bpp: 6,
      stride: w * 6,
      pixels: pixels16,
      extra: [],
    };
    const input = Buffer.from(await writePng(source, deflate));
    expect((await sharp(input).metadata()).depth).toBe('ushort');
    const output = await flipped(input);
    const read = (await readPng(new Uint8Array(output), inflate)) as Png;
    expect(read.bitDepth).toBe(16);
    const planes = rgbPlanes(read);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        expect(Math.round(planes[0][i] * 65535)).toBe(sample(x, y, 0));
        expect(Math.round(planes[1][i] * 65535)).toBe(65535 - sample(x, y, 1));
      }
    }
  });

  test('the preview averages down to the requested size', async () => {
    const { w, h, data } = pixels(3);
    const input = await sharp(data, {
      raw: { width: w, height: h, channels: 3 },
    })
      .png()
      .toBuffer();
    const read = (await readPng(new Uint8Array(input), inflate)) as Png;
    const preview = previewRgba(read, 16);
    expect([preview.width, preview.height]).toEqual([16, 12]);
    expect(preview.data[3]).toBe(255);
  });

  test('a JPEG is not read as PNG', async () => {
    const jpeg = await sharp({
      create: { width: 8, height: 8, channels: 3, background: '#8080ff' },
    })
      .jpeg()
      .toBuffer();
    expect(await readPng(new Uint8Array(jpeg), inflate)).toBeNull();
  });
});
