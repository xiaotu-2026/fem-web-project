// gpu_solver.js — Hybrid GPU/CPU PCG solver
// GPU accelerates SpMV (the O(nnz) bottleneck), CPU handles all
// scalar math and vector updates in Float64 for numerical stability.

const WG_SIZE = 256;

export class GpuJacobiSolver {
    constructor() {
        this.device = null;
        this.ready = false;
        this.gpuUsed = false;
    }

    async init() {
        if (!navigator.gpu) { console.warn('WebGPU not available'); return false; }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { console.warn('No GPU adapter'); return false; }
        try {
            this.device = await adapter.requestDevice();
            const code = await (await fetch('solver.wgsl')).text();
            const module = this.device.createShaderModule({ code });
            const info = await module.getCompilationInfo();
            for (const m of info.messages) {
                if (m.type === 'error') {
                    console.error('WGSL compile error:', m.message);
                    return false;
                }
            }
            const bgl = this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ]
            });
            this.pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });
            this.bgl = bgl;
            this.spmvPipeline = this.device.createComputePipeline({
                layout: this.pipelineLayout,
                compute: { module, entryPoint: 'spmv' }
            });
            this.ready = true;
            console.log('WebGPU PCG solver initialized (hybrid GPU SpMV + CPU f64)');
            return true;
        } catch (e) {
            console.warn('GPU init failed:', e);
            return false;
        }
    }

    _uploadBuffer(usage, data) {
        const buf = this.device.createBuffer({ size: Math.max(data.byteLength, 16), usage, mappedAtCreation: true });
        const view = data instanceof Uint32Array ? new Uint32Array(buf.getMappedRange()) : new Float32Array(buf.getMappedRange());
        view.set(data);
        buf.unmap();
        return buf;
    }

    async solveGPU(rowPtr, colIdx, values, rhs, nDof, maxIter, tol) {
        const d = this.device;
        const nWG = Math.ceil(nDof / WG_SIZE);
        const S = GPUBufferUsage.STORAGE;
        const C = GPUBufferUsage.COPY_SRC;
        const CD = GPUBufferUsage.COPY_DST;

        const bufRowPtr = this._uploadBuffer(S, new Uint32Array(rowPtr));
        const bufColIdx = this._uploadBuffer(S, new Uint32Array(colIdx));
        const bufVals = this._uploadBuffer(S, new Float32Array(values));
        const bufP = this.device.createBuffer({ size: nDof * 4, usage: S | CD });
        const bufAp = this.device.createBuffer({ size: nDof * 4, usage: S | C });
        const bufRead = this.device.createBuffer({ size: nDof * 4, usage: GPUBufferUsage.MAP_READ | CD });
        const paramsArr = new Uint32Array([nDof, 0, 0, 0]);
        const bufParams = this._uploadBuffer(GPUBufferUsage.UNIFORM, paramsArr);

        const bindGroup = d.createBindGroup({
            layout: this.bgl,
            entries: [
                { binding: 0, resource: { buffer: bufRowPtr } },
                { binding: 1, resource: { buffer: bufColIdx } },
                { binding: 2, resource: { buffer: bufVals } },
                { binding: 3, resource: { buffer: bufP } },
                { binding: 4, resource: { buffer: bufAp } },
                { binding: 5, resource: { buffer: bufParams } },
            ]
        });

        const gpuSpMV = async (p64) => {
            const p32 = new Float32Array(nDof);
            for (let i = 0; i < nDof; i++) p32[i] = p64[i];
            d.queue.writeBuffer(bufP, 0, p32);
            const enc = d.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(this.spmvPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(nWG);
            pass.end();
            enc.copyBufferToBuffer(bufAp, 0, bufRead, 0, nDof * 4);
            d.queue.submit([enc.finish()]);
            await bufRead.mapAsync(GPUMapMode.READ);
            const ap32 = new Float32Array(bufRead.getMappedRange().slice(0));
            bufRead.unmap();
            const ap64 = new Float64Array(nDof);
            for (let i = 0; i < nDof; i++) ap64[i] = ap32[i];
            return ap64;
        };

        // PCG with f64 precision, GPU-accelerated SpMV
        const t0 = performance.now();
        const diag = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) {
            for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                if (colIdx[k] === i) { diag[i] = values[k]; break; }
            }
            if (diag[i] === 0) diag[i] = 1.0;
        }
        const x = new Float64Array(nDof);
        const r = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) r[i] = rhs[i];
        const z = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];
        const p = new Float64Array(nDof);
        p.set(z);
        let rz = 0;
        for (let i = 0; i < nDof; i++) rz += r[i] * z[i];
        let bnorm = 0;
        for (let i = 0; i < nDof; i++) bnorm += rhs[i] * rhs[i];
        bnorm = Math.sqrt(bnorm);
        if (bnorm === 0) bnorm = 1.0;

        for (let iter = 0; iter < maxIter; iter++) {
            const Ap = await gpuSpMV(p);
            let pAp = 0;
            for (let i = 0; i < nDof; i++) pAp += p[i] * Ap[i];
            if (Math.abs(pAp) < 1e-300) break;
            const alpha = rz / pAp;
            for (let i = 0; i < nDof; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
            let rnorm = 0;
            for (let i = 0; i < nDof; i++) rnorm += r[i] * r[i];
            rnorm = Math.sqrt(rnorm);
            if (rnorm / bnorm < tol) {
                const dt = ((performance.now() - t0) / 1000).toFixed(3);
                console.log(`GPU PCG converged: iter=${iter + 1}, residual=${(rnorm/bnorm).toExponential(3)}, time=${dt}s`);
                this._cleanup([bufRowPtr, bufColIdx, bufVals, bufP, bufAp, bufRead, bufParams]);
                return x;
            }
            for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];
            let rzNew = 0;
            for (let i = 0; i < nDof; i++) rzNew += r[i] * z[i];
            const beta = rzNew / rz;
            rz = rzNew;
            for (let i = 0; i < nDof; i++) p[i] = z[i] + beta * p[i];
        }
        const dt = ((performance.now() - t0) / 1000).toFixed(3);
        console.log(`GPU PCG reached maxIter=${maxIter}, time=${dt}s`);
        this._cleanup([bufRowPtr, bufColIdx, bufVals, bufP, bufAp, bufRead, bufParams]);
        return x;
    }

    _cleanup(bufs) { bufs.forEach(b => b.destroy()); }

    async solve(rp, ci, v, rhs, n, maxIter = 10000, tol = 1e-6) {
        return this.solveF64(rp, ci, v, rhs, n, maxIter, tol);
    }

    async solveF64(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        if (this.ready) {
            try {
                this.gpuUsed = true;
                return await this.solveGPU(rowPtr, colIdx, values, rhs, nDof, maxIter, tol);
            } catch (e) {
                console.warn('GPU solve failed, falling back to CPU:', e);
            }
        }
        this.gpuUsed = false;
        return this.solveCPU_PCG(rowPtr, colIdx, values, rhs, nDof, maxIter, tol);
    }

    solveCPU_PCG(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        console.log(`CPU PCG: nDof=${nDof}`);
        const t0 = performance.now();
        const diag = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) {
            for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                if (colIdx[k] === i) { diag[i] = values[k]; break; }
            }
            if (diag[i] === 0) diag[i] = 1.0;
        }
        const x = new Float64Array(nDof);
        const r = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) r[i] = rhs[i];
        const z = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];
        const p = new Float64Array(nDof);
        p.set(z);
        const Ap = new Float64Array(nDof);
        let rz = 0;
        for (let i = 0; i < nDof; i++) rz += r[i] * z[i];
        let bnorm = 0;
        for (let i = 0; i < nDof; i++) bnorm += rhs[i] * rhs[i];
        bnorm = Math.sqrt(bnorm);
        if (bnorm === 0) bnorm = 1.0;
        for (let iter = 0; iter < maxIter; iter++) {
            for (let i = 0; i < nDof; i++) {
                let sum = 0;
                for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) sum += values[k] * p[colIdx[k]];
                Ap[i] = sum;
            }
            let pAp = 0;
            for (let i = 0; i < nDof; i++) pAp += p[i] * Ap[i];
            if (Math.abs(pAp) < 1e-300) break;
            const alpha = rz / pAp;
            for (let i = 0; i < nDof; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
            let rnorm = 0;
            for (let i = 0; i < nDof; i++) rnorm += r[i] * r[i];
            rnorm = Math.sqrt(rnorm);
            if (rnorm / bnorm < tol) {
                console.log(`CPU PCG converged: iter=${iter + 1}, residual=${(rnorm/bnorm).toExponential(3)}, time=${((performance.now()-t0)/1000).toFixed(3)}s`);
                return x;
            }
            for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];
            let rzNew = 0;
            for (let i = 0; i < nDof; i++) rzNew += r[i] * z[i];
            const beta = rzNew / rz;
            rz = rzNew;
            for (let i = 0; i < nDof; i++) p[i] = z[i] + beta * p[i];
        }
        console.log(`CPU PCG maxIter=${maxIter}, time=${((performance.now()-t0)/1000).toFixed(3)}s`);
        return x;
    }
}
