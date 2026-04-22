// SpMV compute shader: Ap = A * p (CSR format, f32)

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> pVec: array<f32>;
@group(0) @binding(4) var<storage, read_write> ApVec: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let start = rowPtr[i];
    let end = rowPtr[i + 1u];
    var sum: f32 = 0.0;
    for (var k = start; k < end; k = k + 1u) {
        sum = sum + vals[k] * pVec[colIdx[k]];
    }
    ApVec[i] = sum;
}
