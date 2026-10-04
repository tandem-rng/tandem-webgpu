// Entry point `fill_f32`, appended to tandem.wgsl when the shader is embedded. It reuses that
// file's bindings and building blocks. A separate file keeps tandem.wgsl identical to the copy
// in tandem-rs.
//
// Same traversal as `fill`, but each word w is stored as the float (w >> 8) * 2^-24, the spec's
// Float32 mapping. The shift leaves 24 bits, so the conversion and the power-of-two scaling are
// exact. WGSL has no f64 type, so there is no f64 counterpart: Float64 stays a host mapping.

const F32_SCALE: f32 = 1.0 / 16777216.0;

@compute @workgroup_size(THREADS)
fn fill_f32(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let gi = t >> 3u;
    let lane = t & 7u;
    let g = add64(add64(P.g0, wg.x * GROUPS), gi);
    let c = add64(shl64(g, 3u), lane);
    var s = F_keyed(P.key, c.x, c.y, DOMAIN_STREAM, AUX_STREAM);
    let first = add64(shl64(g, firstTrailingBit(P.K) + 3u), lane);
    for (var j = 0u; j < P.K; j++) {
        s = T(s);
        let i = sub64_small(add64(first, j * 8u), P.base);
        if (i >= 0 && u32(i) < P.n_blocks) {
            out[u32(i)] = bitcast<vec4<u32>>(vec4<f32>(s.o >> vec4<u32>(8u)) * F32_SCALE);
        }
    }
}
