/**
 * Reading and rewriting PNG files byte for byte, without a canvas. A
 * canvas premultiplies alpha (wiping the color under transparent pixels),
 * applies color management and drops 16-bit samples to 8; a normal map
 * needs none of that. Non-interlaced PNGs of 8 or 16 bits per sample are
 * handled; anything else falls back to the browser's decoder.
 */

import type { Deflate } from './encode';
import { chunkBytes, concatBytes } from './encode';

/** Zlib-wrapped inflate (the browser's `DecompressionStream`, zlib in Node). */
export type Inflate = (data: Uint8Array) => Promise<Uint8Array>;

export const browserInflate: Inflate = async (data) => {
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
};

const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

export interface Png {
  width: number;
  height: number;
  /** 8 or 16; palette images also 1, 2 or 4 (packed indices). */
  bitDepth: number;
  /** 0 gray, 2 RGB, 3 palette, 4 gray + alpha, 6 RGBA. */
  colorType: number;
  channels: number;
  /** Bytes per pixel and per row (without the filter byte). */
  bpp: number;
  stride: number;
  /** Unfiltered pixel rows, `height * stride` bytes, big-endian samples. */
  pixels: Uint8Array;
  /** Every chunk except IHDR, IDAT and IEND, in order (palette included). */
  extra: { type: string; data: Uint8Array }[];
}

export const isPng = (bytes: Uint8Array) =>
  SIGNATURE.every((value, i) => bytes[i] === value);

/** Decodes a PNG, or returns null for one this module does not handle. */
export async function readPng(
  bytes: Uint8Array,
  inflate: Inflate
): Promise<Png | null> {
  if (!isPng(bytes)) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 8;
  let header: DataView | null = null;
  const idat: Uint8Array[] = [];
  const extra: Png['extra'] = [];
  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR')
      header = new DataView(data.buffer, data.byteOffset, 13);
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    else extra.push({ type, data });
    offset += 12 + length;
  }
  if (!header) return null;
  const width = header.getUint32(0);
  const height = header.getUint32(4);
  const bitDepth = header.getUint8(8);
  const colorType = header.getUint8(9);
  const interlace = header.getUint8(12);
  const channels = CHANNELS[colorType];
  if (!channels || interlace !== 0) return null;
  const supported =
    colorType === 3
      ? [1, 2, 4, 8].includes(bitDepth)
      : [8, 16].includes(bitDepth);
  if (!supported) return null;
  // Filters work on whole bytes: at least one per pixel.
  const bpp = Math.max(1, (channels * bitDepth) / 8);
  const stride = Math.ceil((width * channels * bitDepth) / 8);
  const raw = await inflate(concatBytes(idat));
  if (raw.length < height * (stride + 1)) return null;
  const pixels = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const out = pixels.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? pixels.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? out[i - bpp] : 0;
      const b = prev ? prev[i] : 0;
      const c = prev && i >= bpp ? prev[i - bpp] : 0;
      let value = line[i];
      if (filter === 1) value += a;
      else if (filter === 2) value += b;
      else if (filter === 3) value += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        value += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      } else if (filter !== 0) return null;
      out[i] = value & 0xff;
    }
  }
  return {
    width,
    height,
    bitDepth,
    colorType,
    channels,
    bpp,
    stride,
    pixels,
    extra,
  };
}

/** The palette (PLTE) of a palette PNG, RGB triples. */
function palette(png: Png): Uint8Array | null {
  return png.extra.find((chunk) => chunk.type === 'PLTE')?.data ?? null;
}

/** Whether the image has color (RGB, RGBA or palette), as normal maps do. */
export const hasColor = (png: Png) => [2, 3, 6].includes(png.colorType);

/**
 * Turns the green channel upside down in place (value to max minus value):
 * every pixel for RGB and RGBA, the palette entries for a palette image.
 * Red, blue, alpha, bit depth and the other chunks stay as they are.
 */
export function flipGreen(png: Png) {
  if (png.colorType === 3) {
    const plte = palette(png);
    if (plte) for (let i = 1; i < plte.length; i += 3) plte[i] = 255 - plte[i];
    return;
  }
  if (!hasColor(png)) return;
  const { pixels, bpp } = png;
  if (png.bitDepth === 8) {
    for (let i = 1; i < pixels.length; i += bpp) pixels[i] = 255 - pixels[i];
  } else {
    for (let i = 2; i < pixels.length; i += bpp) {
      const v = 65535 - ((pixels[i] << 8) | pixels[i + 1]);
      pixels[i] = v >> 8;
      pixels[i + 1] = v & 0xff;
    }
  }
}

/** Writes the PNG back, every row with the Sub filter, all chunks kept. */
export async function writePng(
  png: Png,
  deflate: Deflate
): Promise<Uint8Array> {
  const { width, height, stride, bpp, pixels } = png;
  const raw = new Uint8Array(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    const row = pixels.subarray(y * stride, (y + 1) * stride);
    const o = y * (stride + 1);
    raw[o] = 1; // Sub: each byte minus the one a pixel to the left.
    for (let i = 0; i < stride; i++) {
      raw[o + 1 + i] = (row[i] - (i >= bpp ? row[i - bpp] : 0)) & 0xff;
    }
  }
  const header = new Uint8Array(13);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, width);
  hv.setUint32(4, height);
  header[8] = png.bitDepth;
  header[9] = png.colorType;
  // Ancillary chunks before the image data go first, the rest after it.
  const before = png.extra.filter(
    (c) => !['tEXt', 'zTXt', 'iTXt', 'tIME'].includes(c.type)
  );
  const after = png.extra.filter((c) =>
    ['tEXt', 'zTXt', 'iTXt', 'tIME'].includes(c.type)
  );
  return concatBytes([
    new Uint8Array(SIGNATURE),
    chunkBytes('IHDR', header),
    ...before.map((c) => chunkBytes(c.type, c.data)),
    chunkBytes('IDAT', await deflate(raw)),
    ...after.map((c) => chunkBytes(c.type, c.data)),
    chunkBytes('IEND', new Uint8Array(0)),
  ]);
}

/** One color sample (0..1) of pixel `i`, channel `c` (0 R, 1 G, 2 B). */
function sampleAt(png: Png, plte: Uint8Array | null, i: number, c: number) {
  const { pixels, bpp, colorType, bitDepth } = png;
  if (colorType === 3) {
    let index = pixels[i];
    if (bitDepth < 8) {
      const x = i % png.width;
      const byte =
        pixels[Math.floor(i / png.width) * png.stride + ((x * bitDepth) >> 3)];
      index =
        (byte >> (8 - bitDepth - ((x * bitDepth) & 7))) & ((1 << bitDepth) - 1);
    }
    return (plte?.[index * 3 + c] ?? 0) / 255;
  }
  // Gray images repeat the one channel.
  const channel = colorType === 0 || colorType === 4 ? 0 : c;
  const at = i * bpp + channel * (bitDepth / 8);
  return bitDepth === 8
    ? pixels[at] / 255
    : ((pixels[at] << 8) | pixels[at + 1]) / 65535;
}

/** R, G and B as 0..1 planes, full 16-bit precision. */
export function rgbPlanes(png: Png) {
  const n = png.width * png.height;
  const plte = palette(png);
  const planes = [0, 1, 2].map(() => new Float32Array(n)) as [
    Float32Array,
    Float32Array,
    Float32Array,
  ];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) planes[c][i] = sampleAt(png, plte, i, c);
  }
  return planes;
}

/** A small opaque RGBA copy (box-averaged), for previews and detection. */
export function previewRgba(png: Png, max: number) {
  const { width, height } = png;
  const scale = Math.min(1, max / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const plte = palette(png);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor((y * height) / h);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / h));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor((x * width) / w);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / w));
      // At most a 4x4 grid of samples per output pixel keeps it quick.
      const sy = Math.max(1, Math.floor((y1 - y0) / 4));
      const sx = Math.max(1, Math.floor((x1 - x0) / 4));
      const sum = [0, 0, 0];
      let count = 0;
      for (let yy = y0; yy < y1; yy += sy) {
        for (let xx = x0; xx < x1; xx += sx) {
          const i = yy * width + xx;
          for (let c = 0; c < 3; c++) sum[c] += sampleAt(png, plte, i, c);
          count++;
        }
      }
      const o = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) out[o + c] = (sum[c] / count) * 255 + 0.5;
      out[o + 3] = 255;
    }
  }
  return { width: w, height: h, data: out };
}
