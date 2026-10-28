/**
 * Makes a full map set from one photo or texture.
 *
 *   pnpm example <image> [--out out] [--size 1024] [--seamless]
 *     [--directx] [--strength 1] [--model deepbump256.onnx]
 *
 * Writes normal.png, height.png (16-bit), roughness.png, ao.png and
 * ball.png (a lit material ball to check the result). With `--model`, the
 * surface comes from the DeepBump network (AI photo mode); without it, from
 * the image itself.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { deflateSync } from 'node:zlib';
import * as ort from 'onnxruntime-web';
import sharp from 'sharp';
import {
  channelMean,
  computeMaps,
  DEFAULT_SETTINGS,
  encodeGray16Png,
  fitSize,
  inferNormals,
  makeSeamless,
  resizeRgb,
  rgbFromRgba,
  shadeSphere,
  TILE,
  toRgba,
} from '../src';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    out: { type: 'string', default: 'out' },
    size: { type: 'string', default: '1024' },
    seamless: { type: 'boolean', default: false },
    directx: { type: 'boolean', default: false },
    strength: { type: 'string', default: '1' },
    model: { type: 'string' },
  },
});
const input = positionals[0];
if (!input) {
  console.error('Usage: pnpm example <image> [--out out] [--model file.onnx]');
  process.exit(1);
}
mkdirSync(values.out, { recursive: true });

const { data, info } = await sharp(input)
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });
const fit = fitSize(info.width, info.height, Number(values.size));
let image = resizeRgb(
  rgbFromRgba(data, info.width, info.height),
  fit.width,
  fit.height
);
if (values.seamless) image = makeSeamless(image);
const { width, height } = image;

let aiNormal: Awaited<ReturnType<typeof inferNormals>> | undefined;
if (values.model) {
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(
    readFileSync(values.model)
  );
  aiNormal = await inferNormals(
    channelMean(image),
    width,
    height,
    async (tile) => {
      const tensor = new ort.Tensor('float32', tile, [1, 1, TILE, TILE]);
      const result = await session.run({ [session.inputNames[0]]: tensor });
      return result[session.outputNames[0]].data as Float32Array;
    },
    {
      wrap: values.seamless,
      onProgress: (done, total) =>
        process.stdout.write(`\rAI tiles ${done}/${total}`),
    }
  );
  process.stdout.write('\n');
}

const maps = computeMaps(
  image,
  {
    ...DEFAULT_SETTINGS,
    source: aiNormal ? 'ai' : 'image',
    format: values.directx ? 'directx' : 'opengl',
    strength: Number(values.strength),
    tile: values.seamless,
  },
  aiNormal
);

const png = (rgba: Uint8ClampedArray, w: number, h: number, name: string) =>
  sharp(Buffer.from(rgba.buffer), { raw: { width: w, height: h, channels: 4 } })
    .png()
    .toFile(join(values.out, name));

await png(toRgba(...maps.normal), width, height, 'normal.png');
await png(toRgba(maps.roughness), width, height, 'roughness.png');
await png(toRgba(maps.ao), width, height, 'ao.png');
const ball = 768;
// The ball reads OpenGL normals: flip a DirectX green channel back.
const glMaps = values.directx
  ? {
      ...maps,
      normal: [
        maps.normal[0],
        maps.normal[1].map((g) => 1 - g),
        maps.normal[2],
      ] as typeof maps.normal,
    }
  : maps;
await png(shadeSphere(image, glMaps, ball), ball, ball, 'ball.png');
const height16 = await encodeGray16Png(
  maps.heightMap,
  width,
  height,
  async (bytes) => deflateSync(bytes)
);
writeFileSync(join(values.out, 'height.png'), height16);
console.log(`Wrote ${width}x${height} maps to ${values.out}/`);
