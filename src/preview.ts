/**
 * Live 3D preview in WebGL2: a flat tile lit by a point light that follows
 * the pointer, or a sphere that turns. Uses the same maps the download gets
 * (base color, normal, roughness, AO, height for the sphere's displacement).
 */

export type PreviewShape = 'plane' | 'sphere';

export interface PreviewMaps {
  width: number;
  height: number;
  albedo: Uint8ClampedArray;
  normal: Uint8ClampedArray;
  roughness: Uint8ClampedArray;
  ao: Uint8ClampedArray;
  heightMap: Uint8ClampedArray;
  /** The normal map's green convention; DirectX is flipped back to draw. */
  directx: boolean;
}

const VERTEX = `#version 300 es
in vec3 aPos;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aUv;
uniform mat4 uProj;
uniform mat4 uModel;
uniform float uRepeat;
uniform float uDisplace;
uniform sampler2D uHeight;
out vec3 vPos;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vUv;
void main() {
  vUv = aUv * uRepeat;
  float h = texture(uHeight, vUv).r;
  vec3 pos = aPos + aNormal * (h - 0.5) * uDisplace;
  vec4 world = uModel * vec4(pos, 1.0);
  vPos = world.xyz;
  vNormal = mat3(uModel) * aNormal;
  vTangent = mat3(uModel) * aTangent;
  gl_Position = uProj * world;
}`;

const FRAGMENT = `#version 300 es
precision highp float;
in vec3 vPos;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vUv;
uniform sampler2D uAlbedo;
uniform sampler2D uNormalMap;
uniform sampler2D uRough;
uniform sampler2D uAo;
uniform vec3 uLight;
uniform vec3 uEye;
uniform float uFlipY;
uniform float uBump;
out vec4 outColor;
void main() {
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent - N * dot(N, vTangent));
  vec3 B = cross(N, T);
  vec3 tn = texture(uNormalMap, vUv).rgb * 2.0 - 1.0;
  tn.y *= uFlipY;
  tn.xy *= uBump;
  vec3 n = normalize(mat3(T, B, N) * tn);
  vec3 L = uLight - vPos;
  float dist = length(L);
  L /= dist;
  vec3 V = normalize(uEye - vPos);
  vec3 H = normalize(L + V);
  vec3 albedo = pow(texture(uAlbedo, vUv).rgb, vec3(2.2));
  float rough = texture(uRough, vUv).r;
  float ao = texture(uAo, vUv).r;
  float ndl = max(dot(n, L), 0.0);
  float a = max(0.04, rough * rough);
  float ndh = max(dot(n, H), 0.0);
  float d = ndh * ndh * (a * a - 1.0) + 1.0;
  float spec = (a * a) / (3.14159 * d * d) * 0.04 * ndl;
  float falloff = 2.0 / (1.0 + 0.3 * dist * dist);
  vec3 color = albedo * (0.22 * ao + ndl * falloff * (0.6 + 0.4 * ao))
    + vec3(spec * falloff);
  outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('shader');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? 'shader');
  }
  return shader;
}

interface Mesh {
  vao: WebGLVertexArrayObject;
  count: number;
}

function buildMesh(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  shape: PreviewShape
): Mesh {
  const pos: number[] = [];
  const nor: number[] = [];
  const tan: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  if (shape === 'plane') {
    const seg = 128;
    for (let y = 0; y <= seg; y++) {
      for (let x = 0; x <= seg; x++) {
        const u = x / seg;
        const v = y / seg;
        pos.push(u * 2 - 1, 1 - v * 2, 0);
        nor.push(0, 0, 1);
        tan.push(1, 0, 0);
        uv.push(u, v);
      }
    }
    for (let y = 0; y < seg; y++) {
      for (let x = 0; x < seg; x++) {
        const a = y * (seg + 1) + x;
        idx.push(a, a + seg + 1, a + 1, a + 1, a + seg + 1, a + seg + 2);
      }
    }
  } else {
    const rings = 96;
    const segs = 192;
    for (let r = 0; r <= rings; r++) {
      const theta = (r / rings) * Math.PI;
      for (let s = 0; s <= segs; s++) {
        const phi = (s / segs) * Math.PI * 2;
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.cos(theta);
        const z = -Math.sin(theta) * Math.sin(phi);
        pos.push(x * 0.85, y * 0.85, z * 0.85);
        nor.push(x, y, z);
        tan.push(-Math.sin(phi), 0, -Math.cos(phi));
        // Twice around the equator so the texture keeps its proportions.
        uv.push((s / segs) * 2, r / rings);
      }
    }
    for (let r = 0; r < rings; r++) {
      for (let s = 0; s < segs; s++) {
        const a = r * (segs + 1) + s;
        idx.push(a, a + segs + 1, a + 1, a + 1, a + segs + 1, a + segs + 2);
      }
    }
  }
  const vao = gl.createVertexArray();
  if (!vao) throw new Error('vao');
  gl.bindVertexArray(vao);
  const attrib = (name: string, data: number[], size: number) => {
    const loc = gl.getAttribLocation(program, name);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  };
  attrib('aPos', pos, 3);
  attrib('aNormal', nor, 3);
  attrib('aTangent', tan, 3);
  attrib('aUv', uv, 2);
  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, count: idx.length };
}

function perspective(fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  // Column-major; camera at the origin looking down -z.
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    2 * far * near * nf,
    0,
  ]);
}

function rotation(yaw: number, pitch: number, z: number) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  // R = Rx(pitch) * Ry(yaw), then moved to z.
  return new Float32Array([
    cy,
    sp * sy,
    -cp * sy,
    0,
    0,
    cp,
    sp,
    0,
    sy,
    -sp * cy,
    cp * cy,
    0,
    0,
    0,
    z,
    1,
  ]);
}

export class Preview {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private meshes: Partial<Record<PreviewShape, Mesh>> = {};
  private textures: Record<string, WebGLTexture> = {};
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private frame = 0;
  /** Drawing only while the canvas is on screen and the tab is visible. */
  private onScreen = true;
  private observer: IntersectionObserver | null = null;
  private onVisibility = () => this.resume();
  private hasMaps = false;
  private flipY = 1;
  private start = performance.now();
  shape: PreviewShape = 'sphere';
  repeat = 1;
  /** Pointer in -1..1 (x right, y up); null drifts the light by itself. */
  pointer: [number, number] | null = null;
  yaw = 0;
  pitch = 0.35;
  spinning = true;
  onFirstFrame?: () => void;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('webgl2');
    this.gl = gl;
    const program = gl.createProgram();
    if (!program) throw new Error('program');
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) ?? 'link');
    }
    this.program = program;
    for (const name of [
      'uProj',
      'uModel',
      'uRepeat',
      'uDisplace',
      'uHeight',
      'uAlbedo',
      'uNormalMap',
      'uRough',
      'uAo',
      'uLight',
      'uEye',
      'uFlipY',
      'uBump',
    ]) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    for (const name of ['albedo', 'normal', 'rough', 'ao', 'height']) {
      const texture = gl.createTexture();
      if (!texture) throw new Error('texture');
      this.textures[name] = texture;
    }
    gl.enable(gl.DEPTH_TEST);
    this.loop = this.loop.bind(this);
    this.frame = requestAnimationFrame(this.loop);
    if (typeof IntersectionObserver !== 'undefined') {
      this.observer = new IntersectionObserver((entries) => {
        this.onScreen = entries.some((entry) => entry.isIntersecting);
        this.resume();
      });
      this.observer.observe(canvas);
    }
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  private get active() {
    return this.onScreen && document.visibilityState === 'visible';
  }

  /** Restarts the loop after it stopped off screen. */
  private resume() {
    if (this.active && !this.frame) {
      this.frame = requestAnimationFrame(this.loop);
    }
  }

  setMaps(maps: PreviewMaps) {
    const gl = this.gl;
    const upload = (name: string, data: Uint8ClampedArray) => {
      gl.bindTexture(gl.TEXTURE_2D, this.textures[name]);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        maps.width,
        maps.height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        new Uint8Array(data.buffer, data.byteOffset, data.length)
      );
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_MIN_FILTER,
        gl.LINEAR_MIPMAP_LINEAR
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    };
    upload('albedo', maps.albedo);
    upload('normal', maps.normal);
    upload('rough', maps.roughness);
    upload('ao', maps.ao);
    upload('height', maps.heightMap);
    this.flipY = maps.directx ? -1 : 1;
    this.hasMaps = true;
  }

  private mesh(shape: PreviewShape): Mesh {
    let mesh = this.meshes[shape];
    if (!mesh) {
      mesh = buildMesh(this.gl, this.program, shape);
      this.meshes[shape] = mesh;
    }
    return mesh;
  }

  private loop(now: number) {
    // Off screen or in a background tab: stop until `resume`.
    if (!this.active) {
      this.frame = 0;
      return;
    }
    this.frame = requestAnimationFrame(this.loop);
    if (!this.hasMaps) return;
    const gl = this.gl;
    const canvas = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (w === 0 || h === 0) return;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    const t = (now - this.start) / 1000;
    const plane = this.shape === 'plane';
    // Camera 3.4 units back with a 33 degree view: the plane just fills it.
    const z = plane ? -3.4 : -3.1;
    const fov = (33 * Math.PI) / 180;
    if (!plane && this.spinning) this.yaw += 0.004;
    const model = plane ? rotation(0, 0, z) : rotation(this.yaw, this.pitch, z);
    const [px, py] = this.pointer ?? [
      Math.cos(t * 0.6) * 0.7,
      Math.sin(t * 0.6) * 0.7,
    ];
    const light = plane
      ? [px * 1.1, py * 1.1, z + 0.75]
      : [px * 2.2, py * 2.2, z + 2.2];
    const u = this.uniforms;
    gl.uniformMatrix4fv(u.uProj, false, perspective(fov, w / h, 0.1, 20));
    gl.uniformMatrix4fv(u.uModel, false, model);
    gl.uniform1f(u.uRepeat, this.repeat);
    gl.uniform1f(u.uDisplace, plane ? 0 : 0.035);
    gl.uniform3f(u.uLight, light[0], light[1], light[2]);
    gl.uniform3f(u.uEye, 0, 0, 0);
    gl.uniform1f(u.uFlipY, this.flipY);
    gl.uniform1f(u.uBump, 1);
    const bind = (unit: number, name: string, uniform: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[name]);
      gl.uniform1i(u[uniform], unit);
    };
    bind(0, 'albedo', 'uAlbedo');
    bind(1, 'normal', 'uNormalMap');
    bind(2, 'rough', 'uRough');
    bind(3, 'ao', 'uAo');
    bind(4, 'height', 'uHeight');
    const mesh = this.mesh(this.shape);
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
    if (this.onFirstFrame) {
      this.onFirstFrame();
      this.onFirstFrame = undefined;
    }
  }

  destroy() {
    cancelAnimationFrame(this.frame);
    this.observer?.disconnect();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
