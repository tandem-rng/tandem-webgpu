// The WebGPU fill: stream words in row order from an aligned bit position.

import { align, checkK, DEFAULT_K, type Key, mapF64 } from "./core.ts";
import { SHADER } from "./shader.ts";

export type DType = "u8" | "u16" | "u32" | "u64" | "f32" | "f64";

const WIDTH: Record<DType, number> = { u8: 8, u16: 16, u32: 32, u64: 64, f32: 32, f64: 64 };
const GROUPS = 32;
// WebGPU flag values. TypeScript's DOM lib declares the types but not these constants.
const MAP_READ = 0x1, COPY_SRC = 0x4, COPY_DST = 0x8, UNIFORM = 0x40, STORAGE = 0x80;

export type FillOptions = {
  key: Key;
  /** Stream bit position before alignment. Default 0. */
  position?: bigint;
  count: number;
  dtype: DType;
  /** Chunk length, a power of two in [1, 65536]. Default 32. */
  K?: number;
  /** Write into this storage buffer instead of a new one. It must hold the whole blocks the
   * fill covers; `fillBuffer` reports that size when it allocates. */
  buffer?: GPUBuffer;
  /** For `f32` only: store the spec's floats `(raw >> 8) * 2^-24` instead of raw words, so the
   * buffer holds Float32 values ready for a later GPU stage. Default false. */
  floats?: boolean;
  /** For `u32` and `u64`: bounded integers in [0, range) by Lemire's method, Appendix A of the
   * specification. Element i takes draw i and a rejected draw retries on a fallback generator
   * keyed by the draw's index in the stream, so a fill cut at any element boundary equals the
   * whole fill. A range of 0 gives 0. An empty fill leaves the position unchanged. */
  range?: bigint | number;
  /** For `f32`: standard normals by Box-Muller, Appendix A. Elements 2j and 2j + 1 come from
   * uniform draws 2j and 2j + 1, so an odd count consumes one draw more than it writes, and
   * `buffer` must hold the blocks of that many draws. Computed in single precision with
   * the arithmetic of tandem-c, so values agree with other ports to 16 ulps plus 1e-6. An empty fill leaves the position unchanged. */
  normal?: boolean;
  /** For `f32`: standard exponentials -log(1 - u), one uniform draw per element, with the
   * polynomial logarithm of tandem-c. An empty fill leaves the position unchanged. The values
   * agree with other ports within 8 ulps plus 1e-6, since WGSL does not promise the fused
   * multiply-add. */
  exponential?: boolean;
};

export type Fill<T> = { values: T; position: bigint };

const pipelines = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

type Entry =
  | "fill"
  | "fill_tile"
  | "fill_f32"
  | "fill_below32"
  | "fill_below64"
  | "normal_pairs"
  | "exponential_f32";

function pipelineFor(device: GPUDevice, entryPoint: Entry) {
  let byEntry = pipelines.get(device);
  if (!byEntry) pipelines.set(device, byEntry = new Map());
  let p = byEntry.get(entryPoint);
  if (!p) {
    p = device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint },
    });
    byEntry.set(entryPoint, p);
  }
  return p;
}

/** Request a device with the largest buffer limits the adapter allows. */
export async function requestDevice(adapter?: GPUAdapter | null): Promise<GPUDevice> {
  adapter ??= await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  return adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize, maxBufferSize },
  });
}

type Placed = { buffer: GPUBuffer; byteOffset: number; byteLength: number; position: bigint };
type Job = {
  entry: Entry;
  /** Binding 0, the fill parameters. The pair pass has none. */
  params?: Uint32Array<ArrayBuffer>;
  /** Binding 2, the range or the pair layout. */
  extra?: Uint32Array<ArrayBuffer>;
  workgroups: [number, number];
  buffer: GPUBuffer;
  /** The bound window of `buffer`, never larger than the adapter's binding limit. */
  offset: number;
  size: number;
};

const MAX_WORKGROUPS = 65535;
// Rows per tile of `fill_tile`, which needs K to be a multiple of it.
const TILE_STEPS = 4;

/** Apple GPUs run the tile kernel five times slower than the direct stores, an A100 faster.
 * Browsers name the vendor, and Deno on Metal names the GPU in the description. */
function isApple(device: GPUDevice): boolean {
  const info = device.adapterInfo;
  return /apple/i.test(`${info.vendor} ${info.description}`);
}
const PAIR_THREADS = 256;

/** Largest window one dispatch may bind: the binding limit, and what 65535 workgroups reach. */
function windowBytes(device: GPUDevice, K: number): number {
  const step = Math.max(256, device.limits.minStorageBufferOffsetAlignment);
  const reach = (MAX_WORKGROUPS * GROUPS - 1) * 128 * K;
  return Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, reach) / step) * step;
}

/** Validate one fill, allocate its buffer and compute its dispatches. Touches no queue. */
function plan(
  device: GPUDevice,
  {
    key,
    position = 0n,
    count,
    dtype,
    K = DEFAULT_K,
    buffer,
    floats = false,
    range,
    normal,
    exponential,
  }: FillOptions,
): { placed: Placed; jobs: Job[] } {
  checkK(K);
  const w = WIDTH[dtype];
  if (floats && dtype !== "f32") throw new RangeError("floats applies to dtype f32 only");
  if (normal && (dtype !== "f32" || range !== undefined)) {
    throw new RangeError("normal applies to dtype f32 only, without range");
  }
  if (exponential && (dtype !== "f32" || range !== undefined || normal)) {
    throw new RangeError("exponential applies to dtype f32 only, without range or normal");
  }
  if (range !== undefined && dtype !== "u32" && dtype !== "u64") {
    throw new RangeError("range applies to dtype u32 and u64 only");
  }
  const bound = range === undefined ? 0n : BigInt(range);
  if (bound < 0n || bound >> BigInt(w)) throw new RangeError(`range must fit in ${w} bits`);

  // A normal pair needs two draws, so an odd count still consumes the second one.
  const draws = count + (normal ? count % 2 : 0);
  const p0 = align(position, w);
  const p1 = p0 + BigInt(w) * BigInt(draws);
  const blockStart = p0 >> 7n;
  const blockEnd = (p1 + 127n) >> 7n;
  const nBlocks = Number(blockEnd - blockStart);
  const byteLength = BigInt(w) * BigInt(count) / 8n;
  const byteOffset = Number(p0 - (blockStart << 7n)) / 8;

  if (buffer && buffer.size < nBlocks * 16) {
    throw new RangeError(`buffer holds ${buffer.size} bytes, the fill needs ${nBlocks * 16}`);
  }
  if (!buffer && nBlocks * 16 > device.limits.maxBufferSize) {
    throw new RangeError(`the fill needs ${nBlocks * 16} bytes, over maxBufferSize`);
  }
  buffer ??= device.createBuffer({ size: Math.max(16, nBlocks * 16), usage: STORAGE | COPY_SRC });
  // A derived fill with no elements draws nothing, so it does not even align the position.
  const derived = normal || exponential || range !== undefined;
  const placed = {
    buffer,
    byteOffset,
    byteLength: Number(byteLength),
    position: derived && count === 0 ? position : p1,
  };
  if (nBlocks === 0 || (derived && count === 0)) return { placed, jobs: [] };

  const entry: Entry = range !== undefined
    ? (w === 32 ? "fill_below32" : "fill_below64")
    : floats || normal || exponential
    ? "fill_f32"
    : K % TILE_STEPS === 0 && !isApple(device)
    ? "fill_tile"
    : "fill";
  const extra = range !== undefined
    ? new Uint32Array([Number(bound & 0xffffffffn), Number(bound >> 32n), 0, 0])
    : undefined;

  // Every fill is exact at any block boundary, so the buffer is cut into windows that each fit
  // one binding and one dispatch. A window starts at a multiple of the offset alignment.
  const win = windowBytes(device, K);
  const rowsPerGroup = BigInt(K);
  const jobs: Job[] = [];
  for (let offset = 0; offset < nBlocks * 16; offset += win) {
    const size = Math.min(win, nBlocks * 16 - offset);
    const first = blockStart + BigInt(offset / 16);
    const last = first + BigInt(size / 16) - 1n;
    const g0 = (first >> 3n) / rowsPerGroup;
    const g1 = (last >> 3n) / rowsPerGroup;
    const params = new Uint32Array(12);
    params.set(key, 0);
    params[4] = Number(g0 & 0xffffffffn);
    params[5] = Number(g0 >> 32n);
    params[6] = Number(first & 0xffffffffn);
    params[7] = Number(first >> 32n);
    params[8] = size / 16;
    params[9] = K;
    const workgroups = Math.ceil(Number(g1 - g0 + 1n) / GROUPS);
    jobs.push({ entry, params, extra, workgroups: [workgroups, 1], buffer, offset, size });
  }
  if (exponential) {
    // One element per invocation, laid over the elements after every fill window has written.
    const slot0 = byteOffset / 4, perWindow = win / 4 - 64;
    for (let j0 = 0; j0 < count; j0 += perWindow) {
      const n = Math.min(perWindow, count - j0), start = slot0 + j0;
      const offset = Math.floor(start * 4 / 256) * 256;
      const size = Math.min(win, nBlocks * 16 - offset);
      const groups = Math.ceil(n / PAIR_THREADS);
      const x = Math.min(groups, MAX_WORKGROUPS), y = Math.ceil(groups / x);
      if (y > MAX_WORKGROUPS) throw new RangeError("fill too large for one dispatch");
      jobs.push({
        entry: "exponential_f32",
        extra: new Uint32Array([start - offset / 4, n, n, x]),
        workgroups: [x, y],
        buffer,
        offset,
        size,
      });
    }
    return { placed, jobs };
  }
  if (!normal) return { placed, jobs };

  // Pair chunks start at even elements, so a pair never straddles a window. Pair windows are
  // laid over the elements, after every fill window has written its raw words.
  const slot0 = byteOffset / 4;
  const pairsPerWindow = win / 8 - 32;
  const pairs = Math.ceil(count / 2);
  for (let j0 = 0; j0 < pairs; j0 += pairsPerWindow) {
    const n = Math.min(2 * pairsPerWindow, count - 2 * j0);
    const chunkPairs = Math.ceil(n / 2);
    const start = slot0 + 2 * j0;
    const offset = Math.floor(start * 4 / 256) * 256;
    const size = Math.min(win, nBlocks * 16 - offset);
    const groups = Math.ceil(chunkPairs / PAIR_THREADS);
    const x = Math.min(groups, MAX_WORKGROUPS), y = Math.ceil(groups / x);
    if (y > MAX_WORKGROUPS) throw new RangeError("fill too large for one dispatch");
    jobs.push({
      entry: "normal_pairs",
      extra: new Uint32Array([start - offset / 4, n, chunkPairs, x]),
      workgroups: [x, y],
      buffer,
      offset,
      size,
    });
  }
  return { placed, jobs };
}

/** Encode every job into one compute pass and submit once. */
async function dispatch(device: GPUDevice, jobs: Job[]): Promise<void> {
  if (jobs.length === 0) return;
  const pipelines = new Map<Entry, GPUComputePipeline>();
  for (const { entry } of jobs) {
    if (!pipelines.has(entry)) pipelines.set(entry, await pipelineFor(device, entry));
  }
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  const uniforms: GPUBuffer[] = [];
  const uniform = (data: Uint32Array<ArrayBuffer>) => {
    const u = device.createBuffer({ size: data.byteLength, usage: UNIFORM | COPY_DST });
    device.queue.writeBuffer(u, 0, data);
    uniforms.push(u);
    return u;
  };
  for (const { entry, params, extra, workgroups, buffer, offset, size } of jobs) {
    const pipeline = pipelines.get(entry)!;
    const entries: GPUBindGroupEntry[] = [{ binding: 1, resource: { buffer, offset, size } }];
    if (params) entries.push({ binding: 0, resource: { buffer: uniform(params) } });
    if (extra) entries.push({ binding: 2, resource: { buffer: uniform(extra) } });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }),
    );
    pass.dispatchWorkgroups(...workgroups);
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
  for (const u of uniforms) u.destroy();
}

/**
 * Fill several buffers with one command buffer and one submit, which costs one queue round
 * trip instead of one per fill. Each item is a `fillBuffer` request. Every item is checked
 * before anything is submitted. Items run in order, so two items may share a buffer.
 */
export async function fillMany(
  device: GPUDevice,
  items: readonly FillOptions[],
): Promise<Placed[]> {
  const planned = items.map((item) => plan(device, item));
  await dispatch(device, planned.flatMap(({ jobs }) => jobs));
  return planned.map(({ placed }) => placed);
}

/**
 * Fill `count` values of `dtype` into a new storage buffer, as `count` scalar draws would.
 * The buffer holds whole 16-byte stream blocks from `blockStart`; the values begin at
 * `byteOffset` inside it. Use `fill` for a host array. The words are raw stream words, even
 * for `f32` and `f64`, unless `floats` is set.
 */
export async function fillBuffer(device: GPUDevice, options: FillOptions): Promise<Placed> {
  return (await fillMany(device, [options]))[0];
}

/** Element types `fill` returns. `fillBuffer` takes `DType` only. */
export type HostDType = DType | "bool" | "i8" | "i16" | "i32" | "i64";
type FillRequest<D> = Omit<FillOptions, "floats" | "dtype"> & { dtype: D };

export type Values<D extends HostDType> = D extends "u8" | "bool" ? Uint8Array
  : D extends "u16" ? Uint16Array
  : D extends "u32" ? Uint32Array
  : D extends "u64" ? BigUint64Array
  : D extends "i8" ? Int8Array
  : D extends "i16" ? Int16Array
  : D extends "i32" ? Int32Array
  : D extends "i64" ? BigInt64Array
  : D extends "f32" ? Float32Array
  : Float64Array;

// A signed integer is the unsigned word read two's complement, so the same stream bytes serve.
const SAME_BYTES = { i8: "u8", i16: "u16", i32: "u32", i64: "u64" } as const;

/**
 * Fill on the GPU and read back a typed array of `dtype`, plus the successor position. `bool`
 * returns a Uint8Array of 0 and 1, bit i of the stream from the position.
 */
export async function fill<D extends HostDType>(
  device: GPUDevice,
  options: FillRequest<D>,
): Promise<Fill<Values<D>>> {
  if (options.dtype === "bool") return await fillBool(device, options) as Fill<Values<D>>;
  const dtype = options.dtype in SAME_BYTES
    ? SAME_BYTES[options.dtype as keyof typeof SAME_BYTES]
    : options.dtype as DType;
  const { buffer, byteOffset, byteLength, position } = await fillBuffer(device, {
    ...options,
    dtype,
    floats: dtype === "f32",
  });
  const staging = device.createBuffer({
    size: buffer.size,
    usage: MAP_READ | COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const bytes = staging.getMappedRange().slice(byteOffset, byteOffset + byteLength);
  staging.unmap();
  staging.destroy();
  if (!options.buffer) buffer.destroy();
  return { values: convert(bytes, options.dtype) as Values<D>, position };
}

type Plain = Omit<FillOptions, "dtype" | "floats" | "range" | "normal">;

/** Bounded integers in [0, range) on the GPU, read back. See `FillOptions.range`. */
export function fillBelow<D extends "u32" | "u64">(
  device: GPUDevice,
  options: Plain & { dtype: D; range: bigint | number },
): Promise<Fill<Values<D>>> {
  return fill(device, options);
}

/** Standard normals as Float32 on the GPU, read back. See `FillOptions.normal`. */
export function fillNormal(device: GPUDevice, options: Plain): Promise<Fill<Float32Array>> {
  return fill(device, { ...options, dtype: "f32", normal: true });
}

/** Standard exponentials as Float32 on the GPU, read back. See `FillOptions.exponential`. */
export function fillExponential(device: GPUDevice, options: Plain): Promise<Fill<Float32Array>> {
  return fill(device, { ...options, dtype: "f32", exponential: true });
}

/** Bits are unaligned to bytes, so read whole words that cover them and cut the bits out. */
async function fillBool(
  device: GPUDevice,
  { position = 0n, count, ...rest }: Omit<FillRequest<"bool">, "dtype">,
): Promise<Fill<Uint8Array>> {
  const end = position + BigInt(count);
  const wordStart = position & ~31n;
  const words = Number((end - wordStart + 31n) >> 5n);
  const { values } = await fill(device, {
    ...rest,
    position: wordStart,
    count: words,
    dtype: "u32",
  });
  const skip = Number(position - wordStart);
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) out[i] = (values[(skip + i) >> 5] >>> ((skip + i) & 31)) & 1;
  return { values: out, position: end };
}

function convert(bytes: ArrayBuffer, dtype: HostDType) {
  switch (dtype) {
    case "u8":
      return new Uint8Array(bytes);
    case "u16":
      return new Uint16Array(bytes);
    case "u32":
      return new Uint32Array(bytes);
    case "u64":
      return new BigUint64Array(bytes);
    case "i8":
      return new Int8Array(bytes);
    case "i16":
      return new Int16Array(bytes);
    case "i32":
      return new Int32Array(bytes);
    case "i64":
      return new BigInt64Array(bytes);
    case "f32":
      return new Float32Array(bytes);
    case "f64": {
      const out = new Float64Array(bytes);
      mapF64(out, new Uint32Array(bytes), out.length);
      return out;
    }
  }
}
