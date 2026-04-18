// Jacobi iteration for sparse CSR matrix: x_new[i] = (b[i] - sum(A[i][j]*x[j], j!=i)) / A[i][i]
// Buffers:
//   csrRowPtr: array<u32>  (nDof+1)
//   csrColIdx: array<u32>  (nnz)
//   csrValues: array<f32>  (nnz)
//   rhs:       array<f32>  (nDof)
//   xOld:      array<f32>  (nDof)
//   xNew:      array<f32>  (nDof)

struct Params {
    nDof: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> rhs:    array<f32>;
@group(0) @binding(4) var<storage, read> xOld:   array<f32>;
@group(0) @binding(5) var<storage, read_write> xNew: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.nDof) { return; }

    let start = rowPtr[i];
    let end   = rowPtr[i + 1u];

    var sigma: f32 = 0.0;
    var diag: f32 = 1.0;

    for (var k = start; k < end; k = k + 1u) {
        let j = colIdx[k];
        let v = values[k];
        if (j == i) {
            diag = v;
        } else {
            sigma = sigma + v * xOld[j];
        }
    }

    xNew[i] = (rhs[i] - sigma) / diag;
}
