# tandem-webgpu documentation

- [API](api.md): the CPU class, the GPU fills, the shaders and parallel use.
- [Tests](tests.md): fixtures, hashes and what each suite checks.
- [Speed](speed.md): CPU and GPU figures and rejected variants.

## Install

```sh
npm install github:tandem-rng/tandem-webgpu
```

The install builds `dist/` through the `prepare` script. In a clone, Deno, Bun and Node 24 or
later import `src/mod.ts` directly. The GPU path needs WebGPU. No dependencies.

The stream is bit exact with the specification, and the CPU and GPU return the same values,
except GPU normals, which agree to 16 ulps plus 1e-6, and GPU exponentials, which agree to
8 ulps plus 1e-6.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
