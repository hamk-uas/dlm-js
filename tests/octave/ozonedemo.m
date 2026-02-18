addpath('tests/octave', 'tests/octave/dlm');

%% Real stratospheric ozone data — smoother reference for dlm-js ozone.test.ts
%
% Replicates the model from:
%   Laine, Latva-Pukkila & Kyrölä (2014), ACP 14(18), doi:10.5194/acp-14-9707-2014
%
% Data: tests/ozonedata.dat  (from https://github.com/mjlaine/dlm/tree/master/examples)
%   col 1  decimal year
%   col 2  ozone density [1/cm³]  (NaN where missing)
%   col 3  uncertainty σ [1/cm³]  (NaN where missing)
%   col 4  solar proxy
%   col 5  QBO component 1
%   col 6  QBO component 2
%
% Model: options = struct('trig',2,'order',1) + 3 proxy covariates
%   State layout (m=9): level, slope, 4 trig states, β_solar, β_qbo1, β_qbo2
%
% No MCMC — pure Kalman filter + RTS smoother for deterministic reference.

raw = dlmread('tests/ozonedata.dat');

time  = raw(:,1);
y_raw = raw(:,2);
s_raw = raw(:,3);
X     = raw(:,4:6);   % [N, 3] proxy covariates (always observed)

%% Scale for numerical stability (MATLAB convention: stdnan)
valid_mask = ~isnan(y_raw);
ys = std(y_raw(valid_mask));   % stdnan equivalent — scalar scale factor
yy = y_raw ./ ys;
ss = s_raw ./ ys;

%% Fill missing observations: use mean of valid yy; large sigma to downweight
ym = mean(yy(valid_mask));
ss_valid = ss(valid_mask);
ss_med = median(ss_valid);
yy_filled = yy;
ss_filled = ss;
yy_filled(isnan(yy)) = ym;
ss_filled(isnan(ss)) = ss_med;   % filled sigma (will be dominated by obs in smoother)

%% Process noise (MATLAB ozonedemo defaults)
wtrend = abs(ym) * 0.00005;
wseas  = abs(ym) * 0.015;
w0 = [0, wtrend, wseas, wseas, wseas, wseas];

%% Save inputs
inputs = struct( ...
  'yy',       yy_filled, ...
  'ss',       ss_filled, ...
  'w',        w0,        ...
  'X',        X,         ...
  'ys',       ys,        ...
  'ym',       ym,        ...
  'options',  struct('trig', 2, 'order', 1) ...
);
save_json(inputs, 'tests/ozone-in.json');

%% Run DLM smoother (no MCMC)
options = struct('trig', 2, 'order', 1);
out = dlmfit(yy_filled, ss_filled, w0, [], [], X, options);
save_json(out, 'tests/ozone-out-m.json');

disp('Ozone demo references generated.');
