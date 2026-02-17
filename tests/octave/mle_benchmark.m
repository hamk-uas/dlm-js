addpath('tests/octave', 'tests/octave/dlm');

%% Benchmark MATLAB DLM parameter estimation (fminsearch) on Nile data
% Compares with dlm-js dlmMLE timings in mle-comparison.md

% Load same Nile data used by dlm-js
inputs = load_json("tests/niledemo-in.json");
y = inputs.y;
s = inputs.s;
w = inputs.w;

fprintf('\n=== MATLAB DLM Parameter Estimation Benchmark ===\n');
fprintf('Data: Nile river annual flow (n=%d)\n', length(y));
fprintf('Model: order=1 (local linear trend, m=2)\n\n');

%% 1. Nile with order=1, optimize both s and w (fitv=1, winds=[1,2])
fprintf('--- Test 1: Nile order=1, fit s + w (fitv=1, winds=[1,2]) ---\n');
options1 = struct();
options1.order = 1;
options1.opt = 1;
options1.mcmc = 0;
options1.fitv = 1;
options1.winds = [1, 2];
options1.maxfuneval = 400;
options1.logscale = 1;

% Warm up (first run may include JIT/cache effects in Octave)
out_warmup = dlmfit(y, s, w, [], [], [], options1);

% Timed runs
nruns = 5;
times1 = zeros(nruns, 1);
for i = 1:nruns
  tic;
  out1 = dlmfit(y, s, w, [], [], [], options1);
  times1(i) = toc;
end
fprintf('  Iterations: maxfuneval=%d\n', options1.maxfuneval);
fprintf('  -2logL: %.1f\n', out1.lik);
fprintf('  Wall time: %.1f ms (median of %d runs, after warmup)\n', ...
  median(times1)*1000, nruns);
fprintf('  All runs (ms): ');
fprintf('%.1f ', times1*1000);
fprintf('\n');

% Extract optimized parameters from the result
fprintf('  Optimized V (obs noise): diag = ');
fprintf('%.1f ', diag(out1.V'*out1.V));
fprintf('\n');
fprintf('  Optimized W (state noise): diag = ');
fprintf('%.4f ', diag(out1.W));
fprintf('\n');

%% 2. Nile with order=1, optimize w only (fitv=0, winds=[1,2])
fprintf('\n--- Test 2: Nile order=1, fit w only (fitv=0, winds=[1,2]) ---\n');
options2 = struct();
options2.order = 1;
options2.opt = 1;
options2.mcmc = 0;
options2.fitv = 0;
options2.winds = [1, 2];
options2.maxfuneval = 400;
options2.logscale = 1;

% Warm up
out_warmup2 = dlmfit(y, s, w, [], [], [], options2);

times2 = zeros(nruns, 1);
for i = 1:nruns
  tic;
  out2 = dlmfit(y, s, w, [], [], [], options2);
  times2(i) = toc;
end
fprintf('  Iterations: maxfuneval=%d\n', options2.maxfuneval);
fprintf('  -2logL: %.1f\n', out2.lik);
fprintf('  Wall time: %.1f ms (median of %d runs, after warmup)\n', ...
  median(times2)*1000, nruns);
fprintf('  All runs (ms): ');
fprintf('%.1f ', times2*1000);
fprintf('\n');
fprintf('  Optimized W (state noise): diag = ');
fprintf('%.4f ', diag(out2.W));
fprintf('\n');

%% 3. Nile with order=0 (local level, m=1), optimize s + w
fprintf('\n--- Test 3: Nile order=0, fit s + w (fitv=1, winds=[1]) ---\n');
options3 = struct();
options3.order = 0;
options3.opt = 1;
options3.mcmc = 0;
options3.fitv = 1;
options3.winds = [1];
options3.maxfuneval = 400;
options3.logscale = 1;

% Initial w for order=0 (m=1)
w0 = w(1);

% Warm up
out_warmup3 = dlmfit(y, s, w0, [], [], [], options3);

times3 = zeros(nruns, 1);
for i = 1:nruns
  tic;
  out3 = dlmfit(y, s, w0, [], [], [], options3);
  times3(i) = toc;
end
fprintf('  Iterations: maxfuneval=%d\n', options3.maxfuneval);
fprintf('  -2logL: %.1f\n', out3.lik);
fprintf('  Wall time: %.1f ms (median of %d runs, after warmup)\n', ...
  median(times3)*1000, nruns);
fprintf('  All runs (ms): ');
fprintf('%.1f ', times3*1000);
fprintf('\n');

%% 4. Kaisaniemi seasonal, optimize w
fprintf('\n--- Test 4: Kaisaniemi order=1, trig=1, ns=12, fit w ---\n');
kinputs = load_json("tests/kaisaniemi-in.json");
ky = kinputs.y;
ks = kinputs.s;
kw = kinputs.w;

options4 = struct();
options4.order = 1;
options4.trig = 1;
options4.ns = 12;
options4.opt = 1;
options4.mcmc = 0;
options4.fitv = 0;
options4.winds = [1, 2, 3, 4];
options4.maxfuneval = 800;
options4.logscale = 1;

try
  % Warm up
  out_warmup4 = dlmfit(ky, ks, kw, [], [], [], options4);

  times4 = zeros(nruns, 1);
  for i = 1:nruns
    tic;
    out4 = dlmfit(ky, ks, kw, [], [], [], options4);
    times4(i) = toc;
  end
  fprintf('  Model: m=%d, n=%d\n', size(out4.F,2), length(ky));
  fprintf('  Iterations: maxfuneval=%d\n', options4.maxfuneval);
  fprintf('  -2logL: %.1f\n', out4.lik);
  fprintf('  Wall time: %.1f ms (median of %d runs, after warmup)\n', ...
    median(times4)*1000, nruns);
  fprintf('  All runs (ms): ');
  fprintf('%.1f ', times4*1000);
  fprintf('\n');
  fprintf('  Optimized W (state noise): diag = ');
  fprintf('%.4f ', diag(out4.W));
  fprintf('\n');
catch e
  fprintf('  FAILED: %s\n', e.message);
  fprintf('  (Nelder-Mead often fails to converge for m=4, 4 parameters)\n');
  times4 = [NaN];
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('  Test 1 (Nile, m=2, fit s+w, 3 params): %.1f ms\n', median(times1)*1000);
fprintf('  Test 2 (Nile, m=2, fit w, 2 params):    %.1f ms\n', median(times2)*1000);
fprintf('  Test 3 (Nile, m=1, fit s+w, 2 params):  %.1f ms\n', median(times3)*1000);
fprintf('  Test 4 (Kaisaniemi, m=4, fit w, 4 params): %.1f ms\n', median(times4)*1000);
fprintf('  (median of %d runs each, after 1 warmup run)\n', nruns);
fprintf('\nNote: Octave fminsearch display output (iter/fval) is interleaved above.\n');
fprintf('Suppress with optimset("Display","off") if clean output is needed.\n');
