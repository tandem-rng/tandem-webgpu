// The WebGPU fill: stream words in row order from an aligned bit position.

import { align, DEFAULT_K, type Key, toFloat32, toFloat64 } from "./core.ts";
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
};

export type Fill<T> = { values: T; position: bigint };

const pipelines = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

function pipelineFor(device: GPUDevice): Promise<GPUComputePipeline> {
  let p = pipelines.get(device);
  if (!p) {
    p = device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "fill" },
    });
    pipelines.set(device, p);
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

/**
 * Fill `count` values of `dtype` into a new storage buffer, as `count` scalar draws would.
 * The buffer holds whole 16-byte stream blocks from `blockStart`; the values begin at
 * `byteOffset` inside it. Use `fill` for a host array.
 */
export async function fillBuffer(
  device: GPUDevice,
  { key, position = 0n, count, dtype, K = DEFAULT_K, buffer }: FillOptions,
): Promise<{ buffer: GPUBuffer; byteOffset: number; byteLength: number; position: bigint }> {
  if (!Number.isInteger(Math.log2(K)) || K < 1 || K > 65536) {
    throw new RangeError("K must be a power of two in [1, 65536]");
  }
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
  buffer ??= device.createBuffer({
    size: Math.max(16, nBlocks * 16),
    usage: STORAGE | COPY_SRC,
  });
  if (nBlocks === 0) return { buffer, byteOffset, byteLength, position: p1 };

  const rowsPerGroup = BigInt(K);
  const g0 = (blockStart >> 3n) / rowsPerGroup;
  const g1 = ((blockEnd - 1n) >> 3n) / rowsPerGroup;
  const workgroups = Math.ceil(Number(g1 - g0 + 1n) / GROUPS);
  if (workgroups > 65535) {
    throw new RangeError("fill too large for one dispatch: split the fill by position");
  }

  const params = new Uint32Array(12);
  params.set(key, 0);
  params[4] = Number(g0 & 0xffffffffn);
  params[5] = Number(g0 >> 32n);
  params[6] = Number(blockStart & 0xffffffffn);
  params[7] = Number(blockStart >> 32n);
  params[8] = nBlocks;
  params[9] = K;
  const uniform = device.createBuffer({
    size: params.byteLength,
    usage: UNIFORM | COPY_DST,
  });
  device.queue.writeBuffer(uniform, 0, params);

  const pipeline = await pipelineFor(device);
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  uniform.destroy();
  return { buffer, byteOffset, byteLength, position: p1 };
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
  options: FillOptions & { dtype: D },
): Promise<Fill<Values<D>>> {
  const { buffer, byteOffset, byteLength, position } = await fillBuffer(device, options);
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
      return Float32Array.from(new Uint32Array(bytes), toFloat32);
    case "f64":
      return Float64Array.from(new BigUint64Array(bytes), toFloat64);
  }
}
