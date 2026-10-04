// The WebGPU fill: stream words in row order from an aligned bit position.

import { align, checkK, DEFAULT_K, type Key, toFloat64 } from "./core.ts";
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
};

export type Fill<T> = { values: T; position: bigint };

const pipelines = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

function pipelineFor(device: GPUDevice, entryPoint: "fill" | "fill_f32") {
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
  entry: "fill" | "fill_f32";
  params: Uint32Array<ArrayBuffer>;
  workgroups: number;
  buffer: GPUBuffer;
};

/** Validate one fill, allocate its buffer and compute its dispatch. Touches no queue. */
function plan(
  device: GPUDevice,
  { key, position = 0n, count, dtype, K = DEFAULT_K, buffer, floats = false }: FillOptions,
): { placed: Placed; job?: Job } {
  checkK(K);
  if (floats && dtype !== "f32") throw new RangeError("floats applies to dtype f32 only");
  const w = WIDTH[dtype];
  const p0 = align(position, w);
  const p1 = p0 + BigInt(w) * BigInt(count);
  const blockStart = p0 >> 7n;
  const blockEnd = (p1 + 127n) >> 7n;
  const nBlocks = Number(blockEnd - blockStart);
  const byteLength = Number(p1 - p0) / 8;
  const byteOffset = Number(p0 - (blockStart << 7n)) / 8;

  if (buffer && buffer.size < nBlocks * 16) {
    throw new RangeError(`buffer holds ${buffer.size} bytes, the fill needs ${nBlocks * 16}`);
  }
  const rowsPerGroup = BigInt(K);
  const g0 = (blockStart >> 3n) / rowsPerGroup;
  const g1 = ((blockEnd - 1n) >> 3n) / rowsPerGroup;
  const workgroups = nBlocks === 0 ? 0 : Math.ceil(Number(g1 - g0 + 1n) / GROUPS);
  if (workgroups > 65535) {
    throw new RangeError("fill too large for one dispatch: split the fill by position");
  }
  buffer ??= device.createBuffer({ size: Math.max(16, nBlocks * 16), usage: STORAGE | COPY_SRC });
  const placed = { buffer, byteOffset, byteLength, position: p1 };
  if (nBlocks === 0) return { placed };

  const params = new Uint32Array(12);
  params.set(key, 0);
  params[4] = Number(g0 & 0xffffffffn);
  params[5] = Number(g0 >> 32n);
  params[6] = Number(blockStart & 0xffffffffn);
  params[7] = Number(blockStart >> 32n);
  params[8] = nBlocks;
  params[9] = K;
  return { placed, job: { entry: floats ? "fill_f32" : "fill", params, workgroups, buffer } };
}

/** Encode every job into one compute pass and submit once. */
async function dispatch(device: GPUDevice, jobs: Job[]): Promise<void> {
  if (jobs.length === 0) return;
  const pipelines = new Map<Job["entry"], GPUComputePipeline>();
  for (const { entry } of jobs) {
    if (!pipelines.has(entry)) pipelines.set(entry, await pipelineFor(device, entry));
  }
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  const uniforms = jobs.map(({ entry, params, workgroups, buffer }) => {
    const uniform = device.createBuffer({ size: params.byteLength, usage: UNIFORM | COPY_DST });
    device.queue.writeBuffer(uniform, 0, params);
    const pipeline = pipelines.get(entry)!;
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer } },
        ],
      }),
    );
    pass.dispatchWorkgroups(workgroups);
    return uniform;
  });
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
  await dispatch(device, planned.flatMap(({ job }) => job ?? []));
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

type Values<D extends DType> = D extends "u8" ? Uint8Array
  : D extends "u16" ? Uint16Array
  : D extends "u32" ? Uint32Array
  : D extends "u64" ? BigUint64Array
  : D extends "f32" ? Float32Array
  : Float64Array;

/** Fill on the GPU and read back a typed array of `dtype`, plus the successor position. */
export async function fill<D extends DType>(
  device: GPUDevice,
  options: Omit<FillOptions, "floats"> & { dtype: D },
): Promise<Fill<Values<D>>> {
  const { buffer, byteOffset, byteLength, position } = await fillBuffer(device, {
    ...options,
    floats: options.dtype === "f32",
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
  buffer.destroy();
  return { values: convert(bytes, options.dtype) as Values<D>, position };
}

function convert(bytes: ArrayBuffer, dtype: DType) {
  switch (dtype) {
    case "u8":
      return new Uint8Array(bytes);
    case "u16":
      return new Uint16Array(bytes);
    case "u32":
      return new Uint32Array(bytes);
    case "u64":
      return new BigUint64Array(bytes);
    case "f32":
      return new Float32Array(bytes);
    case "f64":
      return Float64Array.from(new BigUint64Array(bytes), toFloat64);
  }
}
