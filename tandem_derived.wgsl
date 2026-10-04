// Entry points for bounded integers and normals, appended to tandem.wgsl when the shader is
// embedded (see tandem_f32.wgsl). They follow Appendix A of the specification.
//
// `fill_below32` and `fill_below64` map every stream word to Lemire's bounded value. Element i
// is draw i of the plain fill and a rejected draw retries on a fallback generator keyed by the
// global draw index g, so a fill cut at any element boundary equals the whole fill.
//
// `normal_pairs` turns a buffer of f32 uniforms (from `fill_f32`) into Box-Muller pairs in
// place. Pair j is elements 2j and 2j + 1, so the pairs may straddle stream blocks, which a
// separate pass handles without neighbour exchange.

// Second uniform: (range lo, range hi) for the bounded fills, (slot, n, pairs, workgroups in x) for the pairs.
@group(0) @binding(2) var<uniform> Q: vec4<u32>;

const PURPOSE_BELOW_LO32: u32 = 0x4c573332u; // 0x424c573332 is the 32-bit purpose
const PURPOSE_BELOW_LO64: u32 = 0x4c573634u; // 0x424c573634 is the 64-bit purpose
const PURPOSE_BELOW_HI: u32 = 0x42u;

// Word j of the stream of `key` from position 0, for chunk length K (a power of two).
fn stream_word(key: vec4<u32>, j: u32, K: u32) -> u32 {
    let b = j >> 2u;
    let row = b >> 3u;
    let r = block(key, (row / K) * 8u + (b & 7u), 0u, row % K);
    return r[j & 3u];
}

// Fallback generator key: split(g) of purpose(P_w) of the fill's key.
fn fallback_key(key: vec4<u32>, purpose_lo: u32, g: vec2<u32>) -> vec4<u32> {
    return split_key(sub_key(key, purpose_lo, PURPOSE_BELOW_HI), g.x, g.y);
}

fn below32(x: u32, range: u32, key: vec4<u32>, g: vec2<u32>, K: u32) -> u32 {
    if (range == 0u) {
        return 0u;
    }
    if (x * range >= range) {
        return mul_hi(x, range);
    }
    let t = (0u - range) % range;
    if (x * range >= t) {
        return mul_hi(x, range);
    }
    let k2 = fallback_key(key, PURPOSE_BELOW_LO32, g);
    for (var j = 0u; ; j++) {
        let y = stream_word(k2, j, K);
        if (y * range >= t) {
            return mul_hi(y, range);
        }
    }
    return 0u;
}

// ---- 64-bit helpers on (lo, hi) pairs ---------------------------------------------------

fn lt64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

// The 128-bit product a * b as (low 64 bits, high 64 bits).
fn mul64(a: vec2<u32>, b: vec2<u32>) -> array<vec2<u32>, 2> {
    let l00 = a.x * b.x;
    let h00 = mul_hi(a.x, b.x);
    let l01 = a.x * b.y;
    let h01 = mul_hi(a.x, b.y);
    let l10 = a.y * b.x;
    let h10 = mul_hi(a.y, b.x);
    let l11 = a.y * b.y;
    let h11 = mul_hi(a.y, b.y);
    let s1 = h00 + l01;
    let s2 = s1 + l10;
    let c1 = select(0u, 1u, s1 < h00) + select(0u, 1u, s2 < s1);
    let u1 = h01 + h10;
    let u2 = u1 + l11;
    let u3 = u2 + c1;
    let c2 = select(0u, 1u, u1 < h01) + select(0u, 1u, u2 < u1) + select(0u, 1u, u3 < u2);
    return array<vec2<u32>, 2>(vec2<u32>(l00, s2), vec2<u32>(u3, h11 + c2));
}

// a mod n by binary long division. Runs only when a draw lands in the rare low band.
fn mod64(a: vec2<u32>, n: vec2<u32>) -> vec2<u32> {
    var r = vec2<u32>(0u, 0u);
    for (var i = 63i; i >= 0i; i--) {
        let bit = select((a.x >> u32(i)) & 1u, (a.y >> u32(i - 32i)) & 1u, i >= 32i);
        let over = (r.y >> 31u) == 1u;
        r = vec2<u32>((r.x << 1u) | bit, (r.y << 1u) | (r.x >> 31u));
        if (over || !lt64(r, n)) {
            r = sub64(r, n);
        }
    }
    return r;
}

fn below64(x: vec2<u32>, range: vec2<u32>, key: vec4<u32>, g: vec2<u32>, K: u32) -> vec2<u32> {
    if (range.x == 0u && range.y == 0u) {
        return vec2<u32>(0u, 0u);
    }
    var m = mul64(x, range);
    if (!lt64(m[0], range)) {
        return m[1];
    }
    let t = mod64(sub64(vec2<u32>(0u, 0u), range), range);
    if (!lt64(m[0], t)) {
        return m[1];
    }
    let k2 = fallback_key(key, PURPOSE_BELOW_LO64, g);
    for (var j = 0u; ; j++) {
        let y = vec2<u32>(stream_word(k2, 2u * j, K), stream_word(k2, 2u * j + 1u, K));
        m = mul64(y, range);
        if (!lt64(m[0], t)) {
            return m[1];
        }
    }
    return vec2<u32>(0u, 0u);
}

// ---- Bounded fills -------------------------------------------------------------------------
//
// The traversal is `fill`'s. Stream block bi holds draws 4 bi .. 4 bi + 3 of 32 bits and
// 2 bi, 2 bi + 1 of 64 bits, so bi gives the global draw index g without a start offset.

@compute @workgroup_size(THREADS)
fn fill_below32(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let gi = t >> 3u;
    let lane = t & 7u;
    let g = add64(add64(P.g0, wg.x * GROUPS), gi);
    let c = add64(shl64(g, 3u), lane);
    var s = F_keyed(P.key, c.x, c.y, DOMAIN_STREAM, AUX_STREAM);
    let first = add64(shl64(g, firstTrailingBit(P.K) + 3u), lane);
    for (var j = 0u; j < P.K; j++) {
        s = T(s);
        let bi = add64(first, j * 8u);
        let i = sub64_small(bi, P.base);
        if (i >= 0 && u32(i) < P.n_blocks) {
            let d = shl64(bi, 2u);
            out[u32(i)] = vec4<u32>(
                below32(s.o.x, Q.x, P.key, d, P.K),
                below32(s.o.y, Q.x, P.key, add64(d, 1u), P.K),
                below32(s.o.z, Q.x, P.key, add64(d, 2u), P.K),
                below32(s.o.w, Q.x, P.key, add64(d, 3u), P.K));
        }
    }
}

@compute @workgroup_size(THREADS)
fn fill_below64(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let gi = t >> 3u;
    let lane = t & 7u;
    let g = add64(add64(P.g0, wg.x * GROUPS), gi);
    let c = add64(shl64(g, 3u), lane);
    var s = F_keyed(P.key, c.x, c.y, DOMAIN_STREAM, AUX_STREAM);
    let first = add64(shl64(g, firstTrailingBit(P.K) + 3u), lane);
    let range = vec2<u32>(Q.x, Q.y);
    for (var j = 0u; j < P.K; j++) {
        s = T(s);
        let bi = add64(first, j * 8u);
        let i = sub64_small(bi, P.base);
        if (i >= 0 && u32(i) < P.n_blocks) {
            let d = shl64(bi, 1u);
            let a = below64(s.o.xy, range, P.key, d, P.K);
            let b = below64(s.o.zw, range, P.key, add64(d, 1u), P.K);
            out[u32(i)] = vec4<u32>(a, b);
        }
    }
}

// ---- Normals -------------------------------------------------------------------------------

const TWO_PI_HI: f32 = 6.2831855;
const TWO_PI_LO: f32 = -1.7484555e-7;

// Box-Muller of the uniforms (a, b) in single precision, the arithmetic of tandem-c. WGSL
// leaves the accuracy of log, cos and sin to the implementation, and a software rasteriser
// misses the tolerance, so the two functions are short series that need only fma and sqrt.
//
// ln(1 - a): 1 - a is exact and in (0, 1]. Split it as m 2^e with m in [sqrt(1/2), sqrt(2))
// from its exponent bits, then ln m = 2 s (1 + z/3 + z^2/5 + ...) with s = (m - 1) / (m + 1)
// and z = s^2 <= 0.03.
//
// cos and sin of 2 pi b: b - q/4 for the nearest quarter turn q is exact, so the angle lies in
// [-pi/4, pi/4] and Taylor series give cos and sin there. The quarter turn is a swap and a
// sign change.
fn normal_pair(a: f32, b: f32) -> vec2<f32> {
    let bits = bitcast<u32>(1.0 - a) + 0x004afb0du;
    let nk = f32(127i - i32(bits >> 23u));
    let m = bitcast<f32>((bits & 0x007fffffu) + 0x3f3504f3u);
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let p = fma(z, fma(z, fma(z, 0.14275366, 0.20000061), 0.33333334), 1.0);
    let r = sqrt(fma(nk, 1.38629150390625, (s * -4.0) * p) + nk * 2.857213530660374e-06);

    let q = u32(b * 4.0 + 0.5);
    let f = fma(-f32(q), 0.25, b);
    // 2 pi as a float pair, so the angle is good to the last bit of the float.
    let th = fma(f, TWO_PI_LO, f * TWO_PI_HI);
    let w = th * th;
    let hs = fma(w, fma(w, fma(w, 2.72499e-06, -0.00019840087), 0.008333332), -0.16666667);
    let hc = fma(w, fma(w, fma(w, 2.4463761e-05, -0.0013887589), 0.04166665), -0.5);
    let sn = th * fma(w, hs, 1.0);
    let cs = fma(w, hc, 1.0);
    let k = q & 3u;
    let x = select(select(select(cs, sn, k == 3u), -cs, k == 2u), -sn, k == 1u);
    let y = select(select(select(sn, -cs, k == 3u), -sn, k == 2u), cs, k == 1u);
    return r * vec2<f32>(x, y);
}

// The same buffer as `out`, seen as words. A component store through `out` may rewrite the
// whole vec4 and race with the neighbouring pair, so this pass writes single words.
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;

fn load_f32(slot: u32) -> f32 {
    return bitcast<f32>(out_words[slot]);
}

fn store_f32(slot: u32, v: f32) {
    out_words[slot] = bitcast<u32>(v);
}

// One invocation per pair. The pairs are numbered over a 2-D dispatch, since one dimension
// holds only 65535 workgroups.
@compute @workgroup_size(THREADS)
fn normal_pairs(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.y * Q.w * THREADS + id.x;
    if (j >= Q.z) {
        return;
    }
    let slot = Q.x + 2u * j;
    let z = normal_pair(load_f32(slot), load_f32(slot + 1u));
    store_f32(slot, z.x);
    if (2u * j + 1u < Q.y) {
        store_f32(slot + 1u, z.y);
    }
}
