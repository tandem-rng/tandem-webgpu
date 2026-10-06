<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .wgsl"></p>

# tandem-webgpu

[![CI](https://github.com/tandem-rng/tandem-webgpu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-webgpu/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-webgpu/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[Tandem8x32](https://github.com/tandem-rng/spec) for JavaScript: a WGSL shader fills the stream
on the GPU, and the same package runs every draw on the CPU for Node, Deno, Bun and browsers.
The stream is bit exact with the specification. CPU exponentials and Float32 normals agree with
tandem-c to 4 ulps, GPU normals with the CPU to 16 ulps plus 1e-6 and GPU exponentials to 8 ulps
plus 1e-6.

The git install builds `dist/` through `prepare`. In a clone, Deno, Bun and Node 24 or later
import `src/mod.ts` directly. The GPU path needs WebGPU.

```sh
npm install github:tandem-rng/tandem-webgpu
```

```ts
import { fill, requestDevice, seed, split, Tandem } from "tandem-webgpu";

const key = seed(42n);
const rng = new Tandem(key);
const xs = rng.fillF64(1 << 20);          // CPU fill, bit exact with the specification
const worker = split(key, 7n);            // also sub(purpose), fork(n)
const z = new Tandem(worker).fillNormalF64(1000);
const gpu = await fill(await requestDevice(), { key, count: 1000, dtype: "u32" });
```

See [API](docs/api.md), [tests](docs/tests.md) and [speed](docs/speed.md) for the detail.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-webgpu/) · [Apache 2.0 license](LICENSE)
