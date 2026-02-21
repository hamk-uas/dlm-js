addpath('tests/octave', 'tests/octave/dlm');

%% Gapped data (NaN) test
%
% Tests that the Kalman filter correctly handles missing observations by
% skipping the measurement update at NaN timesteps, matching MATLAB dlmsmo
% behaviour: ig = not(isnan(y(i,:))).
%
% Two sub-tests on Nile data:
%   Test A — order=1 (local level + slope), ~23% missing observations
%   Test B — order=0 (local level only),    ~23% missing observations
%
% Gap pattern is deterministic: every 7th observation is NaN,
% plus a fixed block (indices 30..39).  No MCMC sampling — dlmsmo called
% with sample=0 to avoid mvnorrnan issues with NaN data.

nile_in = load_json("tests/niledemo-in.json");
y_full = nile_in.y;
n = length(y_full);
s = nile_in.s;
w = nile_in.w;

%% Build deterministic NaN mask (reproducible, no random seed needed)
nan_mask = false(n, 1);
nan_mask(7:7:n) = true;          % every 7th observation
nan_mask(30:39) = true;           % contiguous block

y_gapped = y_full;
y_gapped(nan_mask) = NaN;

nobs_expected = n - sum(nan_mask);   % observations actually used
fprintf('n=%d, nans=%d, nobs=%d\n', n, sum(nan_mask), nobs_expected);

%% Helper: two-pass DLM smoother (no MCMC, NaN-safe)
%  Mirrors the logic of dlmfit but calls dlmsmo with sample=0
function out = run_dlm_nosample(y, s, wdiag, options)
  [G, F] = dlmgensys(options);
  [p, m] = size(F);
  n = length(y);
  V = ones(n, p) .* s;
  W = zeros(m, m);
  for i = 1:length(wdiag)
    W(i,i) = wdiag(i).^2;
  end
  % Initial state (NaN-safe mean)
  ns = 12;
  y_init = y(1:min(ns,n));
  y_valid = y_init(~isnan(y_init));
  if isempty(y_valid)
    x0_val = 0;
  else
    x0_val = mean(y_valid);
  end
  x0 = zeros(m, 1);
  x0(1) = x0_val;
  c0 = (abs(x0_val)*0.5)^2;
  if c0 == 0; c0 = 1e7; end
  C0 = diag(ones(1,m) * c0);
  % Pass 1 (sample=0)
  out1 = dlmsmo(y, F, V, x0, G, W, C0, [], 0);
  x0 = out1.x(:,1);
  C0 = 100*squeeze(out1.C(:,:,1));
  % Pass 2 (sample=0 — avoids mvnorrnan with NaN data)
  out = dlmsmo(y, F, V, x0, G, W, C0, [], 0);
end

%% Test A: order=1 (local level + slope, m=2)
fprintf('\nTest A: order=1 with gapped data\n');

options_A = struct('order', 1);
out_A = run_dlm_nosample(y_gapped, s, w, options_A);

fprintf('  nobs=%d (expected %d)\n', out_A.nobs, nobs_expected);

inputs_A = struct( ...
  'y',              y_gapped, ...
  'y_full',         y_full, ...
  'nan_mask',       double(nan_mask), ...
  's',              s, ...
  'w',              w, ...
  'options',        options_A, ...
  'nobs_expected',  nobs_expected ...
);
save_json(inputs_A, 'tests/gapped-in.json');
save_json(out_A,    'tests/gappedout-m.json');
disp('  Saved tests/gapped-{in,out-m}.json');

%% Test B: order=0 (local level only, m=1)
fprintf('\nTest B: order=0 with gapped data\n');

w_level = w(1);
options_B = struct('order', 0);
out_B = run_dlm_nosample(y_gapped, s, w_level, options_B);

fprintf('  nobs=%d (expected %d)\n', out_B.nobs, nobs_expected);

inputs_B = struct( ...
  'y',       y_gapped, ...
  's',       s, ...
  'w',       w_level, ...
  'options', options_B ...
);
save_json(inputs_B, 'tests/gapped-order0-in.json');
save_json(out_B,    'tests/gapped-order0-out-m.json');
disp('  Saved tests/gapped-order0-{in,out-m}.json');

disp('gappeddata_test.m done.');

