// gpu_solver.js — PCG solver (Float64) with GPU Jacobi option

export class GpuJacobiSolver {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.ready = false;
    }

    async init() {
        if (!navigator.gpu) { console.warn('WebGPU not available, using CPU PCG solver'); return false; }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { console.warn('No GPU adapter found'); return false; }
        try {
            this.device = await adapter.requestDevice();
            const shaderCode = await (await fetch('solver.wgsl')).text();
            const shaderModule = this.device.createShaderModule({ code: shaderCode });
            this.pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: shaderModule, entryPoint: 'main' }
            });
            this.ready = true;
            return true;
        } catch (e) {
            console.warn('GPU init failed:', e);
            return false;
        }
    }

    // Main solve entry — always use CPU PCG for reliability
    async solve(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        return this.solveCPU_PCG(rowPtr, colIdx, values, rhs, nDof, maxIter, tol);
    }

    // Solve with Float64 data directly (preferred path)
    async solveF64(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        return this.solveCPU_PCG(rowPtr, colIdx, values, rhs, nDof, maxIter, tol);
    }

    // Preconditioned Conjugate Gradient solver (Diagonal/Jacobi preconditioner)
    // Works with any typed array input, computes in Float64
    solveCPU_PCG(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        console.log(`PCG solver: nDof=${nDof}, maxIter=${maxIter}, tol=${tol}`);
        const t0 = performance.now();

        // Extract diagonal for preconditioner
        const diag = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) {
            for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                if (colIdx[k] === i) {
                    diag[i] = values[k];
                    break;
                }
            }
            if (diag[i] === 0) diag[i] = 1.0; // safety
        }

        // x = 0 initial guess
        const x = new Float64Array(nDof);
        // r = b - A*x = b (since x=0)
        const r = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) r[i] = rhs[i];

        // z = M^{-1} r (diagonal preconditioner)
        const z = new Float64Array(nDof);
        for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];

        // p = z
        const p = new Float64Array(nDof);
        p.set(z);

        // Ap buffer
        const Ap = new Float64Array(nDof);

        let rz = 0;
        for (let i = 0; i < nDof; i++) rz += r[i] * z[i];

        // Compute initial residual norm for relative tolerance
        let bnorm = 0;
        for (let i = 0; i < nDof; i++) bnorm += rhs[i] * rhs[i];
        bnorm = Math.sqrt(bnorm);
        if (bnorm === 0) bnorm = 1.0;

        for (let iter = 0; iter < maxIter; iter++) {
            // Ap = A * p
            for (let i = 0; i < nDof; i++) {
                let sum = 0;
                for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                    sum += values[k] * p[colIdx[k]];
                }
                Ap[i] = sum;
            }

            // alpha = rz / (p^T Ap)
            let pAp = 0;
            for (let i = 0; i < nDof; i++) pAp += p[i] * Ap[i];
            if (Math.abs(pAp) < 1e-300) {
                console.log(`PCG: pAp near zero at iter ${iter}, stopping`);
                break;
            }
            const alpha = rz / pAp;

            // x = x + alpha * p
            // r = r - alpha * Ap
            for (let i = 0; i < nDof; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            // Check convergence
            let rnorm = 0;
            for (let i = 0; i < nDof; i++) rnorm += r[i] * r[i];
            rnorm = Math.sqrt(rnorm);

            if (rnorm / bnorm < tol) {
                const dt = ((performance.now() - t0) / 1000).toFixed(3);
                console.log(`PCG converged at iteration ${iter + 1}, relative residual=${(rnorm/bnorm).toExponential(3)}, time=${dt}s`);
                return x;
            }

            // z = M^{-1} r
            for (let i = 0; i < nDof; i++) z[i] = r[i] / diag[i];

            // beta = (r_new^T z_new) / (r_old^T z_old)
            let rzNew = 0;
            for (let i = 0; i < nDof; i++) rzNew += r[i] * z[i];
            const beta = rzNew / rz;
            rz = rzNew;

            // p = z + beta * p
            for (let i = 0; i < nDof; i++) {
                p[i] = z[i] + beta * p[i];
            }
        }

        const dt = ((performance.now() - t0) / 1000).toFixed(3);
        console.log(`PCG reached maxIter=${maxIter}, time=${dt}s`);
        return x;
    }

    // Legacy CPU Jacobi (kept for reference, not used)
    solveCPU(rowPtr, colIdx, values, rhs, nDof, maxIter = 10000, tol = 1e-6) {
        let x = new Float64Array(nDof);
        let xNew = new Float64Array(nDof);
        for (let iter = 0; iter < maxIter; iter++) {
            let maxDiff = 0;
            for (let i = 0; i < nDof; i++) {
                let sigma = 0, diag = 1;
                for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                    const j = colIdx[k];
                    const v = values[k];
                    if (j === i) diag = v;
                    else sigma += v * x[j];
                }
                xNew[i] = (rhs[i] - sigma) / diag;
                const diff = Math.abs(xNew[i] - x[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            [x, xNew] = [xNew, x];
            if (maxDiff < tol) {
                console.log(`CPU Jacobi converged at iteration ${iter+1}`);
                return x;
            }
        }
        console.log(`CPU Jacobi finished ${maxIter} iterations`);
        return x;
    }
}
