/**
 * DLM Fit Function
 * @param {Object} jaxInstance - Jax-js instance
 * @param {Array} y - Observed data array
 * @param {number} s - Observation noise standard deviation (scalar)
 * @param {Array} w - State noise standard deviations (array of two elements)
 * @returns {Object} - Fitted DLM results
 */
const dlmFit = async (jaxInstance, y, s, w) => {
    const jax = jaxInstance;
    const np = jax.numpy;

    /**
     * DLM Smoother Function
     * @param {Array} y - Observed data array
     * @param {Object} F - Observation matrix (jax-js array)
     * @param {Array} V_std - Observation noise standard deviations (array)
     * @param {Array} x0_data - Initial state mean (2D array)
     * @param {Object} G - State transition matrix (jax-js array)
     * @param {Object} W - State noise covariance matrix (jax-js array)
     * @param {Array} C0_data - Initial state covariance (2D array)
     * @param {boolean} sample - Whether to perform sampling (not implemented here)
     * @returns {Object} - Smoothed DLM results
     */
    const dlmSmo = async (y, F, V_std, x0_data, G, W, C0_data, sample = false) => {
        const n = y.length;

        // Initialize prediction arrays (these are one-step-ahead predictions)
        const x_pred = new Array(n);  // Prediction means
        const C_pred = new Array(n);  // Prediction covariances
        const K_array = new Array(n); // Kalman gains
        const v_array = new Array(n); // Innovations
        const Cp_array = new Array(n); // Innovation covariances

        // Initial state is the initial prediction for time 0
        const x0 = np.array(x0_data, { dtype: "float32" });
        const C0 = np.array(C0_data, { dtype: "float32" });

        // Store initial prediction (copy the data)
        const x0_data_copy = await x0.ref.data();
        const C0_data_copy = await C0.ref.data();
        x_pred[0] = np.array([[x0_data_copy[0]], [x0_data_copy[1]]], { dtype: "float32" });
        C_pred[0] = np.array([[C0_data_copy[0], C0_data_copy[1]], [C0_data_copy[2], C0_data_copy[3]]], { dtype: "float32" });

        // Dispose initial arrays as we've copied them
        x0.dispose();
        C0.dispose();

        // --- Forward Kalman Filter ---
        for (let i = 0; i < n; i++) {
            // Compute innovation: v = y - F*x (where x is prediction)
            const yt = np.array([y[i]], { dtype: "float32" });
            const Fx = np.matmul(F.ref, x_pred[i].ref);
            const v = np.subtract(yt, Fx);
            const v_data = await v.ref.data();
            v_array[i] = v_data[0];

            // Compute innovation covariance: Cp = F*C*F' + V
            const FC = np.matmul(F.ref, C_pred[i].ref);
            const Ft = np.transpose(F.ref);
            const FCFt = np.matmul(FC, Ft);
            const V_val = V_std[i] * V_std[i];
            const V_arr = np.array([[V_val]], { dtype: "float32" });
            const Cp = np.add(FCFt, V_arr);
            const Cp_data = await Cp.ref.data();
            Cp_array[i] = Cp_data[0];

            // Compute Kalman gain: K = G*C*F'/Cp
            const GC = np.matmul(G.ref, C_pred[i].ref);
            const Ft2 = np.transpose(F.ref);
            const GCFt = np.matmul(GC, Ft2);
            // Use jax-js's solve for 1x1 Cp
            const K = np.matmul(GCFt, np.linalg.inv(Cp));
            const K_data = await K.ref.data();
            K_array[i] = np.array([[K_data[0]], [K_data[1]]], { dtype: "float32" });

            // Compute next prediction if not last timestep
            if (i < n - 1) {
                // L = G - K*F
                const KF = np.matmul(K.ref, F.ref);
                const L = np.subtract(G.ref, KF.ref);

                // x(i+1) = G*x(i) + K*v
                const Gx = np.matmul(G.ref, x_pred[i].ref);
                const v_arr = np.array([[v_data[0]]], { dtype: "float32" });
                const Kv = np.matmul(K.ref, v_arr.ref);
                const x_next = np.add(Gx, Kv);

                // C(i+1) = G*C(i)*L' + W
                const GC2 = np.matmul(G.ref, C_pred[i].ref);
                const Lt = np.transpose(L.ref);
                const GCLt = np.matmul(GC2, Lt);
                let C_next = np.add(GCLt, W.ref);

                // No symmetry fix needed with 32-bit and jax-js
                const C_next_data = await C_next.ref.data();

                x_pred[i + 1] = np.array([[x_next.ref.dataSync()[0]], [x_next.ref.dataSync()[1]]], { dtype: "float32" });
                C_pred[i + 1] = np.array([[C_next_data[0], C_next_data[1]], [C_next_data[2], C_next_data[3]]], { dtype: "float32" });

                x_next.dispose();
                C_next.dispose();
                L.dispose();
                KF.dispose();
                v_arr.dispose();
            }

            K.dispose();
            v.dispose();
        }

        // --- Backward Smoother ---
        // Initialize r and N for backward recursion
        let r = np.array([[0.0], [0.0]], { dtype: "float32" });
        let N = np.array([[0.0, 0.0], [0.0, 0.0]], { dtype: "float32" });

        const x_smooth = new Array(n);
        const C_smooth = new Array(n);

        // Backward pass
        for (let i = n - 1; i >= 0; i--) {
            // L = G - K*F
            const KF = np.matmul(K_array[i].ref, F.ref);
            const L = np.subtract(G.ref, KF.ref);

            // Compute F'/Cp where F=[1,0] and Cp is scalar
            // F'/Cp = [[1/Cp], [0]]
            const FFCp_data = [[1.0 / Cp_array[i]], [0.0]];
            const FFCp = np.array(FFCp_data, { dtype: "float32" });

            // r = FFCp*v + L'*r
            const FFCp_v = np.multiply(FFCp.ref, v_array[i]);
            const Lt = np.transpose(L.ref);
            const Ltr = np.matmul(Lt.ref, r.ref);
            const r_new = np.add(FFCp_v, Ltr);

            // N = FFCp*F + L'*N*L
            const FFCp_F = np.matmul(FFCp, F.ref);
            const LtN = np.matmul(Lt, N.ref);
            const LtNL = np.matmul(LtN, L);
            const N_new = np.add(FFCp_F, LtNL);

            // x_smooth = x_pred + C*r
            const Cr = np.matmul(C_pred[i].ref, r_new.ref);
            const x_s = np.add(x_pred[i].ref, Cr.ref);

            // C_smooth = C_pred - C*N*C
            const CN = np.matmul(C_pred[i].ref, N_new.ref);
            const CNC = np.matmul(CN.ref, C_pred[i].ref);
            let C_s = np.subtract(C_pred[i].ref, CNC.ref);

            // No symmetry fix needed with 32-bit and jax-js
            const x_s_data = await x_s.ref.data();
            const C_s_data = await C_s.ref.data();

            x_smooth[i] = np.array([[x_s_data[0]], [x_s_data[1]]], { dtype: "float32" });
            C_smooth[i] = np.array([[C_s_data[0], C_s_data[1]], [C_s_data[2], C_s_data[3]]], { dtype: "float32" });

            // Update r and N for next iteration
            r.dispose();
            N.dispose();
            r = r_new;
            N = N_new;

            // Cleanup
            x_s.dispose();
            C_s.dispose();
            Cr.dispose();
            CN.dispose();
            CNC.dispose();
            KF.dispose();
        }

        r.dispose();
        N.dispose();

        // Don't cleanup x_pred and C_pred arrays - we return them as filtered values!
        // Only cleanup K_array
        for (let i = 0; i < n; i++) {
            K_array[i].dispose();
        }

        // ...existing code...

        // Compute additional outputs to match Octave's dlmsmo output
        const yhat = [];
        const ystd = [];
        const xstd = [];
        const resid0 = [];
        const resid = [];
        const resid2 = [];
        const v_out = [];
        const Cp_out = [];

        let ssy = 0;
        let lik = 0;
        let nobs = n;

        for (let i = 0; i < n; i++) {
            // yhat: filter prediction = F * x_pred
            const yhat_i_data = await x_pred[i].ref.data();
            const yhat_val = yhat_i_data[0]; // F = [1, 0], so F*x = x[0]
            yhat.push(yhat_val);

            // xstd: smoothed state standard deviations
            const C_smooth_data = await x_smooth[i].ref.data();
            const C_data = await C_smooth[i].ref.data();
            xstd.push([Math.sqrt(Math.abs(C_data[0])), Math.sqrt(Math.abs(C_data[3]))]);

            // ystd: prediction standard deviation = sqrt(F*C*F' + V^2)
            const FC_val = C_data[0]; // F*C for F=[1,0] is just C[0,0]
            const ystd_val = Math.sqrt(FC_val + V_std[i] * V_std[i]);
            ystd.push(ystd_val);

            // resid0: raw residuals = y - yhat
            const resid0_val = y[i] - yhat_val;
            resid0.push(resid0_val);

            // resid: scaled residuals = resid0 / V
            const resid_val = resid0_val / V_std[i];
            resid.push(resid_val);

            // ssy: sum of squared raw residuals
            ssy += resid0_val * resid0_val;

            // v: innovations (already computed)
            v_out.push(v_array[i]);

            // Cp: innovation covariances (already computed)
            Cp_out.push(Cp_array[i]);

            // resid2: standardized prediction residuals = v / sqrt(Cp)
            const resid2_val = v_array[i] / Math.sqrt(Cp_array[i]);
            resid2.push(resid2_val);

            // lik: -2*log likelihood for single series
            lik += (v_array[i] * v_array[i]) / Cp_array[i] + Math.log(Cp_array[i]);
        }

        // s2: residual variance
        const s2 = resid.reduce((sum, r) => sum + r * r, 0) / nobs;

        // mse: mean squared error of standardized residuals
        const mse = resid2.reduce((sum, r) => sum + r * r, 0) / nobs;

        // mape: mean absolute percentage error
        const mape = resid2.reduce((sum, r, i) => sum + Math.abs(r) / Math.abs(y[i]), 0) / nobs;

        return {
            x: x_smooth,      // Smoothed states
            C: C_smooth,      // Smoothed covariances
            xf: x_pred,       // Filtered (prediction) states
            Cf: C_pred,       // Filtered (prediction) covariances
            yhat,             // Filter predictions
            ystd,             // Prediction standard deviations
            xstd,             // Smoothed state standard deviations
            resid0,           // Raw residuals
            resid,            // Scaled residuals
            resid2,           // Standardized residuals
            v: v_out,         // Innovations
            Cp: Cp_out,       // Innovation covariances
            ssy,              // Sum of squared residuals
            s2,               // Residual variance
            nobs,             // Number of observations
            lik,              // Likelihood
            mse,              // Mean squared error
            mape              // Mean absolute percentage error
        };
    };


    const n = y.length;

    const G_data = [[1.0, 1.0], [0.0, 1.0]];
    const F_data = [[1.0, 0.0]];
    const G = np.array(G_data, { dtype: "float32" });
    const F = np.array(F_data, { dtype: "float32" });

    const V_std = new Array(n).fill(s);
    const W_data = [[w[0] * w[0], 0.0], [0.0, w[1] * w[1]]];
    const W = np.array(W_data, { dtype: "float32" });

    // Initial Values (Octave dlmfit default init)
    let sum = 0;
    const ns = 12;
    for (let i = 0; i < ns; i++) sum += y[i];
    const mean_y = sum / ns;

    const x0_data = [[mean_y], [0.0]];
    const c0_val = Math.pow(Math.abs(mean_y) * 0.5, 2);
    const safe_c0 = c0_val === 0 ? 1e7 : c0_val;
    const C0_data = [[safe_c0, 0.0], [0.0, safe_c0]];

    // --- Run 1 ---
    const out1 = await dlmSmo(y, F.ref, V_std, x0_data, G.ref, W.ref, C0_data);

    // --- Update Initial Values ---
    const x0_arr = out1.x[0];
    const x0_new_flat = await x0_arr.ref.data();
    const x0_new_data = [x0_new_flat[0], x0_new_flat[1]];

    const C0_arr = out1.C[0];
    const C0_new_flat = await C0_arr.ref.data();
    const C0_scaled_data = [
        [C0_new_flat[0] * 100, C0_new_flat[1] * 100],
        [C0_new_flat[2] * 100, C0_new_flat[3] * 100]
    ];

    // Dispose Run 1 results (only dispose array elements)
    for (let i = 0; i < n; i++) {
        out1.x[i].dispose();
        out1.C[i].dispose();
        out1.xf[i].dispose();
        out1.Cf[i].dispose();
    }

    // --- Run 2 ---
    const out2 = await dlmSmo(y, F, V_std, x0_new_data, G, W, C0_scaled_data, false);

    // --- Format Output ---
    // Return filtered values (xf, Cf) plus all other computed values
    const xf_js = [[], []];
    const Cf_js = [
        [[], []],  // Row 0: [C(0,0) over time, C(0,1) over time]
        [[], []]   // Row 1: [C(1,0) over time, C(1,1) over time]
    ];
    const x_js = [[], []];
    const C_js = [
        [[], []],
        [[], []]
    ];
    const xstd_js = [[], []];

    for (let i = 0; i < n; i++) {
        // Filtered states
        const xfi_jax = out2.xf[i];
        const xfi_flat = await xfi_jax.ref.data();
        xf_js[0].push(xfi_flat[0]);
        xf_js[1].push(xfi_flat[1]);
        xfi_jax.dispose();

        // Filtered covariances
        const Cfi_jax = out2.Cf[i];
        const Cfi_flat = await Cfi_jax.ref.data();
        Cf_js[0][0].push(Cfi_flat[0]);
        Cf_js[0][1].push(Cfi_flat[1]);
        Cf_js[1][0].push(Cfi_flat[2]);
        Cf_js[1][1].push(Cfi_flat[3]);
        Cfi_jax.dispose();

        // Smoothed states
        const xi_jax = out2.x[i];
        const xi_flat = await xi_jax.ref.data();
        x_js[0].push(xi_flat[0]);
        x_js[1].push(xi_flat[1]);
        xi_jax.dispose();

        // Smoothed covariances
        const Ci_jax = out2.C[i];
        const Ci_flat = await Ci_jax.ref.data();
        C_js[0][0].push(Ci_flat[0]);
        C_js[0][1].push(Ci_flat[1]);
        C_js[1][0].push(Ci_flat[2]);
        C_js[1][1].push(Ci_flat[3]);
        Ci_jax.dispose();

        // State standard deviations (transposed)
        xstd_js[i] = [out2.xstd[i][0], out2.xstd[i][1]];
    }

    // Get system matrices as plain arrays
    const G_data_js = await G.ref.data();
    const F_data_js = await F.ref.data();
    const W_data_js = await W.ref.data();

    return {
        xf: xf_js,
        Cf: Cf_js,
        x: x_js,
        C: C_js,
        xstd: xstd_js,
        G: [[G_data_js[0], G_data_js[1]], [G_data_js[2], G_data_js[3]]],
        F: [F_data_js[0], F_data_js[1]],
        W: [[W_data_js[0], W_data_js[1]], [W_data_js[2], W_data_js[3]]],
        y: y,
        V: V_std,
        x0: x0_new_data,
        C0: C0_scaled_data,
        XX: [],  // No covariates
        yhat: out2.yhat,
        ystd: out2.ystd,
        resid0: out2.resid0,
        resid: out2.resid,
        ssy: out2.ssy,
        v: out2.v,
        Cp: out2.Cp,
        s2: out2.s2,
        nobs: out2.nobs,
        lik: out2.lik,
        mse: out2.mse,
        mape: out2.mape,
        resid2: out2.resid2,
        class: 'dlmfit',
        // Note: xrd, xr, xrp, yrp, ss would require sampling implementation
    };
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { dlmFit };
}