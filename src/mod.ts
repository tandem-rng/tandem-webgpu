// tandem-webgpu: Tandem8x32 for WebGPU. Keys and small draws on the CPU, fills on the GPU.
// Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

export {
  align,
  block,
  checkK,
  DEFAULT_K,
  F,
  fKeyed,
  fork,
  seed,
  split,
  sub,
  T,
  Tandem,
  toFloat32,
  toFloat64,
} from "./core.ts";
export type { Key, State } from "./core.ts";
export { fill, fillBuffer, requestDevice } from "./gpu.ts";
export type { DType, Fill, FillOptions } from "./gpu.ts";
