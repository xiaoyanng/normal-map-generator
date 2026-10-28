/**
 * File output without libraries: 16-bit grayscale PNG (deflate comes from
 * the caller, `CompressionStream` in browsers, zlib in Node) and a stored
 * (uncompressed) ZIP for downloading every map at once. PNG data is already
 * compressed, so storing loses nothing.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array, crc = 0xffffffff): number {
  let c = crc;
  for (let i = 0; i < data.length; i++) {
    c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  }
  return c;
}

function crcOf(data: Uint8Array): number {
  return (crc32(data) ^ 0xffffffff) >>> 0;
}

/** Zlib-wrapped deflate, as PNG's IDAT chunk needs. */
export type Deflate = (data: Uint8Array) => Promise<Uint8Array>;

export const browserDeflate: Deflate = async (data) => {
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
};

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crcOf(out.subarray(4, 8 + data.length)));
  return out;
}

/** 16-bit grayscale PNG from a 0..1 plane: no stair steps in displacement. */
export async function encodeGray16Png(
  data: Float32Array,
  width: number,
  height: number,
  deflate: Deflate
): Promise<Uint8Array> {
  const stride = width * 2 + 1;
  const raw = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    // Filter type 0 (none) per row.
    raw[y * stride] = 0;
    for (let x = 0; x < width; x++) {
      const v = Math.round(
        Math.min(1, Math.max(0, data[y * width + x])) * 65535
      );
      raw[y * stride + 1 + x * 2] = v >> 8;
      raw[y * stride + 2 + x * 2] = v & 0xff;
    }
  }
  const header = new Uint8Array(13);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, width);
  hv.setUint32(4, height);
  header[8] = 16; // bit depth
  header[9] = 0; // grayscale
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', await deflate(raw)),
    chunk('IEND', new Uint8Array(0)),
  ];
  return concat(parts);
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((sum, p) => sum + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/** A ZIP of the given files, stored without compression. */
export function makeZip(
  files: { name: string; data: Uint8Array }[]
): Uint8Array {
  const encoder = new TextEncoder();
  const locals: Uint8Array[] = [];
  const centrals: Uint8Array[] = [];
  let offset = 0;
  for (const file of files) {
    const name = encoder.encode(file.name);
    const crc = crcOf(file.data);
    const local = new Uint8Array(30 + name.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(8, 0, true); // stored
    lv.setUint32(14, crc, true);
    lv.setUint32(18, file.data.length, true);
    lv.setUint32(22, file.data.length, true);
    lv.setUint16(26, name.length, true);
    local.set(name, 30);
    const central = new Uint8Array(46 + name.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, file.data.length, true);
    cv.setUint32(24, file.data.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true);
    central.set(name, 46);
    locals.push(local, file.data);
    centrals.push(central);
    offset += local.length + file.data.length;
  }
  const centralSize = centrals.reduce((sum, c) => sum + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  return concat([...locals, ...centrals, end]);
}
