// Tandem8x32 in WGSL: the building blocks of https://github.com/tandem-rng/spec and a
// row fill. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
//
// WGSL has no 64-bit integers. Chunk and row indices travel as (lo, hi) pairs, and the
// 32x32 -> 64 product of the mix is built from 16-bit halves, which is exact.

const CLOCK_WEYL: u32 = 0x9e3779b9u;
const DOMAIN_STREAM: u32 = 0x9e3779b9u;
const DOMAIN_SPLIT: u32 = 0xbb67ae85u;
const DOMAIN_FOLD: u32 = 0xcd9e8d57u;
const AUX_STREAM: u32 = 0x94d049bbu;

fn rotl(x: u32, r: u32) -> u32 {
    return (x << r) | (x >> (32u - r));
}

// High word of the 64-bit product a * b.
fn mul_hi(a: u32, b: u32) -> u32 {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let lh = al * bh;
    let hl = ah * bl;
    let mid = ((al * bl) >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}

struct State {
    o: vec4<u32>,
    h: vec4<u32>,
}

// The step T: mix, clock, feedback.
fn T(s: State) -> State {
    let m0 = s.h.x | 1u;
    let m1 = s.h.y | 1u;
    let lo0 = s.o.x * m0;
    let hi0 = mul_hi(s.o.x, m0);
    let lo1 = s.o.z * m1;
    let hi1 = mul_hi(s.o.z, m1);
    let n = vec4<u32>(s.o.y ^ hi1 ^ lo1, rotl(lo1, 16u) ^ s.h.z, s.o.w ^ hi0 ^ lo0,
                      rotl(lo0, 16u) ^ s.h.w);
    var h = s.h;
    h.x ^= rotl(h.y, 7u);
    h.y ^= rotl(h.z, 13u);
    h.z ^= rotl(h.w, 22u);
    h.w ^= rotl(h.x, 3u);
    h.x = (h.x + CLOCK_WEYL) ^ n.x;
    return State(n, h);
}

fn round(s0: State, rc: u32) -> State {
    let s = T(s0);
    return State(s.h, vec4<u32>(s.o.x ^ rc, s.o.yzw));
}

// The seeding function F: eight rounds of T, a round constant, a half swap. Written out,
// since an array of constants indexed in a loop lands in per-thread memory on some GPUs.
fn F(s: State) -> State {
    var r = round(s, 0xd17cc1b7u);
    r = round(r, 0xa7220a94u);
    r = round(r, 0xfe13abe8u);
    r = round(r, 0xfa9a6ee0u);
    r = round(r, 0xedb14accu);
    r = round(r, 0x9e21c820u);
    r = round(r, 0xff28b1d5u);
    return round(r, 0xef5de2b0u);
}

fn F_keyed(key: vec4<u32>, counter_lo: u32, counter_hi: u32, domain: u32, aux: u32) -> State {
    return F(State(vec4<u32>(counter_lo, counter_hi, domain, aux), key));
}

// Block B(c, j): the exposed half of chunk c after j + 1 steps.
fn block(key: vec4<u32>, c_lo: u32, c_hi: u32, j: u32) -> vec4<u32> {
    var s = F_keyed(key, c_lo, c_hi, DOMAIN_STREAM, AUX_STREAM);
    for (var i = 0u; i <= j; i++) {
        s = T(s);
    }
    return s.o;
}

fn split_key(key: vec4<u32>, index_lo: u32, index_hi: u32) -> vec4<u32> {
    let s = F_keyed(key, (index_lo >> 1u) | (index_hi << 31u), index_hi >> 1u, DOMAIN_SPLIT, 0u);
    return select(s.o, s.h, (index_lo & 1u) == 1u);
}

fn sub_key(key: vec4<u32>, purpose_lo: u32, purpose_hi: u32) -> vec4<u32> {
    return F_keyed(key, purpose_lo, purpose_hi, DOMAIN_FOLD, 0u).o;
}

// ---- Row fill -------------------------------------------------------------------------
//
// One invocation per chunk, 32 groups of 8 lanes per workgroup. `fill` stores each 16-byte
// block straight to the output: the eight lanes of a group write one contiguous 128-byte row.
// `fill_tile` stages TILE_STEPS rows of every group in workgroup memory first, so 32
// consecutive invocations store 512 contiguous bytes. The tile is 4 % faster on an NVIDIA
// A100 under Vulkan and five times slower on an Apple M4 Pro under Metal.

const THREADS = 256u;
const GROUPS = 32u;

struct Params {
    key: vec4<u32>,
    g0: vec2<u32>,       // first group (lo, hi)
    base: vec2<u32>,     // first 16-byte block of the output buffer, as a stream block index
    n_blocks: u32,       // length of the output buffer in 16-byte blocks
    K: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>;

fn add64(a: vec2<u32>, b: u32) -> vec2<u32> {
    let lo = a.x + b;
    return vec2<u32>(lo, a.y + select(0u, 1u, lo < a.x));
}

// (a << s) for 0 < s < 32, a < 2^(64 - s).
fn shl64(a: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}

// a - b as a signed offset, assuming |a - b| < 2^31. Returns -1 when below.
fn sub64_small(a: vec2<u32>, b: vec2<u32>) -> i32 {
    if (a.y < b.y || (a.y == b.y && a.x < b.x)) {
        return -1;
    }
    return i32(a.x - b.x);
}

@compute @workgroup_size(THREADS)
fn fill(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let gi = t >> 3u;
    let lane = t & 7u;
    let g = add64(add64(P.g0, wg.x * GROUPS), gi);
    let c = add64(shl64(g, 3u), lane);
    var s = F_keyed(P.key, c.x, c.y, DOMAIN_STREAM, AUX_STREAM);
    // Stream block index of this chunk's first block: g * K * 8 + lane.
    let first = add64(shl64(g, firstTrailingBit(P.K) + 3u), lane);
    for (var j = 0u; j < P.K; j++) {
        s = T(s);
        let i = sub64_small(add64(first, j * 8u), P.base);
        if (i >= 0 && u32(i) < P.n_blocks) {
            out[u32(i)] = s.o;
        }
    }
}

// 16 KiB of workgroup memory, the WebGPU default limit. Eight steps measured no faster.
const TILE_STEPS = 4u;
var<workgroup> tile: array<vec4<u32>, 1024>;

// The fill through the tile, for K a multiple of TILE_STEPS. A group's TILE_STEPS rows are
// contiguous in the stream, so slot k of the tile is block k of its group's run.
@compute @workgroup_size(THREADS)
fn fill_tile(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let gi = t >> 3u;
    let lane = t & 7u;
    let gw = add64(P.g0, wg.x * GROUPS);
    let c = add64(shl64(add64(gw, gi), 3u), lane);
    var s = F_keyed(P.key, c.x, c.y, DOMAIN_STREAM, AUX_STREAM);
    // The workgroup's first block relative to the buffer's. Fills stay below 2^31 blocks, so
    // the offsets fit an i32.
    let w0 = i32(shl64(gw, firstTrailingBit(P.K) + 3u).x - P.base.x);
    for (var jb = 0u; jb < P.K; jb += TILE_STEPS) {
        for (var j = 0u; j < TILE_STEPS; j++) {
            s = T(s);
            tile[(gi * TILE_STEPS + j) * 8u + lane] = s.o;
        }
        workgroupBarrier();
        for (var k = 0u; k < TILE_STEPS; k++) {
            let slot = t + THREADS * k;
            let i = w0 + i32((slot / (TILE_STEPS * 8u)) * P.K * 8u + jb * 8u + slot % (TILE_STEPS * 8u));
            if (i >= 0 && u32(i) < P.n_blocks) {
                out[u32(i)] = tile[slot];
            }
        }
        workgroupBarrier();
    }
}
