addpath('tests/octave', 'tests/octave/dlm');

%% Test 1: order=0 (local level model) on Nile data
% Simplest DLM: y(t) = level(t) + v, level(t) = level(t-1) + w
% State dimension m = 1

disp('Test 1: order=0 (local level)');
nile_in = load_json("tests/niledemo-in.json");
inputs_o0 = struct("y", nile_in.y, "s", nile_in.s, "w", nile_in.w(1));
save_json(inputs_o0, "tests/order0-in.json");

options_o0 = struct("order", 0);
out_o0 = dlmfit(nile_in.y, nile_in.s, nile_in.w(1), [], [], [], options_o0);
save_json(out_o0, "tests/order0-out-m.json");
disp('  done');

%% Test 2: order=2 (quadratic trend) on synthetic accelerating data
% y(t) = level(t) + v
% level(t) = level(t-1) + slope(t-1) + w1
% slope(t) = slope(t-1) + accel(t-1) + w2
% accel(t) = accel(t-1) + w3
% State dimension m = 3

disp('Test 2: order=2 (quadratic trend)');
n = 100;
% Deterministic quadratic signal + fixed noise pattern
t = (1:n)';
signal = 50 + 0.5*t + 0.01*t.^2;
% Use reproducible "noise" - simple deterministic pattern
noise = 5 * sin((1:n)' * 0.7) + 3 * cos((1:n)' * 1.3);
y_o2 = signal + noise;

s_o2 = 8;
w_o2 = [2, 1, 0.5];

inputs_o2 = struct("y", y_o2, "s", s_o2, "w", w_o2);
save_json(inputs_o2, "tests/order2-in.json");

options_o2 = struct("order", 2);
out_o2 = dlmfit(y_o2, s_o2, w_o2, [], [], [], options_o2);
save_json(out_o2, "tests/order2-out-m.json");
disp('  done');

%% Test 3: fullseas=1 with ns=12 on synthetic monthly data
% Local level + full seasonal (11 seasonal states)
% State dimension m = 2 + 11 = 13

disp('Test 3: fullseas=1, ns=12 (seasonal)');
n_seas = 120;  % 10 years of monthly data
t_seas = (1:n_seas)';
trend = 100 + 0.3*t_seas;
seasonal = 15*sin(2*pi*t_seas/12) + 8*cos(2*pi*2*t_seas/12) + 4*sin(2*pi*3*t_seas/12);
noise_seas = 3 * sin(t_seas * 0.37) + 2 * cos(t_seas * 0.91);
y_seas = trend + seasonal + noise_seas;

s_seas = 5;
% w: [level_noise, slope_noise] — seasonal states have zero noise
w_seas = [3, 0.5];

inputs_seas = struct("y", y_seas, "s", s_seas, "w", w_seas);
save_json(inputs_seas, "tests/seasonal-in.json");

options_seas = struct("order", 1, "fullseas", 1, "ns", 12);
out_seas = dlmfit(y_seas, s_seas, w_seas, [], [], [], options_seas);
save_json(out_seas, "tests/seasonal-out-m.json");
disp('  done');

%% Test 4: trig=2 with ns=12 on same seasonal data
% Local level + 2 trigonometric harmonics (4 seasonal states)
% State dimension m = 2 + 4 = 6

disp('Test 4: trig=2, ns=12 (trigonometric seasonal)');
options_trig = struct("order", 1, "trig", 2, "ns", 12);
inputs_trig = struct("y", y_seas, "s", s_seas, "w", w_seas);
save_json(inputs_trig, "tests/trig-in.json");

out_trig = dlmfit(y_seas, s_seas, w_seas, [], [], [], options_trig);
save_json(out_trig, "tests/trig-out-m.json");
disp('  done');

%% Test 5: order=0 local level only (simplest possible model)
% This tests m=1 specifically — good edge case
disp('Test 5: order=0 minimal (m=1)');
% Use first 50 values of Nile data for a compact test
y_mini = nile_in.y(1:50);
s_mini = sqrt(15100);
w_mini = sqrt(755);

inputs_mini = struct("y", y_mini, "s", s_mini, "w", w_mini);
save_json(inputs_mini, "tests/level-in.json");

options_mini = struct("order", 0);
out_mini = dlmfit(y_mini, s_mini, w_mini, [], [], [], options_mini);
save_json(out_mini, "tests/level-out-m.json");
disp('  done');

%% Test 6: trig=1 with ns=12 + AR(1) on same seasonal data
% Local level+slope + 1 trigonometric harmonic (2 seasonal states) + AR(1)
% State dimension m = 2 + 2 + 1 = 5
% This is the first test combining seasonal and autoregression.

disp('Test 6: trig=1, ns=12, arphi=0.7 (seasonal + AR)');
options_trigar = struct("order", 1, "trig", 1, "ns", 12, "arphi", 0.7);
% w: [level, slope, trig_cos, trig_sin, ar]
w_trigar = [3, 0.5, 0.4, 0.4, 1.0];
inputs_trigar = struct("y", y_seas, "s", s_seas, "w", w_trigar);
save_json(inputs_trigar, "tests/trigar-in.json");

out_trigar = dlmfit(y_seas, s_seas, w_trigar, [], [], [], options_trigar);
save_json(out_trigar, "tests/trigar-out-m.json");
disp('  done');

%% Test 7: Synthetic energy demand — trend + seasonal + AR(1)
% Simulates monthly energy consumption with:
%   - Linear growth trend (rising demand)
%   - Seasonal cycle (1 trig harmonic, ns=12)
%   - AR(1) with phi=0.425 (moderate weather/economic deviations)
%   - Observation noise
% Data generated from the DLM state-space model itself using a fixed seed.
% State dimension m = 2 + 2 + 1 = 5

disp('Test 7: synthetic energy demand (trend + seasonal + AR)');

% Generate system matrices
options_energy = struct("order", 1, "trig", 1, "ns", 12, "arphi", 0.425);
[G_e, F_e] = dlmgensys(options_energy);
m_e = size(G_e, 1);  % should be 5

% Noise parameters (standard deviations)
% w: [level, slope, trig_cos, trig_sin, ar_state]
w_energy = [0.3, 0.02, 0.15, 0.15, 2.5];
s_energy = 1.5;

% Build W (process noise covariance) and V (obs noise variance)
W_e = zeros(m_e, m_e);
for i = 1:length(w_energy)
  W_e(i,i) = w_energy(i)^2;
end
V_e = s_energy^2;

% Generate data from the model with fixed seed
rng(42, 'twister');
n_e = 120;  % 10 years monthly

% Initial state: [level=100, slope=0.2, cos=0, sin=0, ar=0]
x_true = zeros(m_e, n_e);
y_energy = zeros(n_e, 1);
x_prev = [100; 0.2; 0; 0; 0];

for t_i = 1:n_e
  % State transition + process noise
  w_noise = zeros(m_e, 1);
  for j = 1:m_e
    w_noise(j) = w_energy(min(j, length(w_energy))) * randn();
  end
  x_t = G_e * x_prev + w_noise;
  x_true(:, t_i) = x_t;

  % Observation + observation noise
  y_energy(t_i) = F_e * x_t + s_energy * randn();
  x_prev = x_t;
end

inputs_energy = struct("y", y_energy, "s", s_energy, "w", w_energy);
save_json(inputs_energy, "tests/energy-in.json");

out_energy = dlmfit(y_energy, s_energy, w_energy, [], [], [], options_energy);
% Also save the true states for reference
out_energy.x_true = x_true;
save_json(out_energy, "tests/energy-out-m.json");
disp('  done');

%% Test 8: Synthetic data with AR(2) — trend + damped oscillatory AR
% Simulates monthly data with:
%   - Linear trend (order=1)
%   - AR(2) with phi=[0.6, -0.3] (damped oscillation, complex roots)
%   - No seasonal component (isolates AR(2) behavior)
%   - Observation noise
% Data generated from the DLM state-space model itself using a fixed seed.
% State dimension m = 2 + 2 = 4 (trend + AR companion block)

disp('Test 8: synthetic AR(2) (damped oscillation)');

% Generate system matrices
options_ar2 = struct("order", 1, "arphi", [0.6, -0.3]);
[G_a2, F_a2] = dlmgensys(options_ar2);
m_a2 = size(G_a2, 1);  % should be 4

% Noise parameters (standard deviations)
% w: [level, slope, ar1_state, ar2_state]
w_ar2 = [0.5, 0.05, 2.0, 0];
s_ar2 = 1.0;

% Build W (process noise covariance)
W_a2 = zeros(m_a2, m_a2);
for i = 1:length(w_ar2)
  W_a2(i,i) = w_ar2(i)^2;
end

% Generate data from the model with fixed seed
rng(99, 'twister');
n_a2 = 100;

% Initial state: [level=50, slope=0.1, ar1=0, ar2=0]
x_true_a2 = zeros(m_a2, n_a2);
y_ar2 = zeros(n_a2, 1);
x_prev_a2 = [50; 0.1; 0; 0];

for t_i = 1:n_a2
  % State transition + process noise
  w_noise = zeros(m_a2, 1);
  for j = 1:m_a2
    w_noise(j) = w_ar2(min(j, length(w_ar2))) * randn();
  end
  x_t = G_a2 * x_prev_a2 + w_noise;
  x_true_a2(:, t_i) = x_t;

  % Observation + observation noise
  y_ar2(t_i) = F_a2 * x_t + s_ar2 * randn();
  x_prev_a2 = x_t;
end

inputs_ar2 = struct("y", y_ar2, "s", s_ar2, "w", w_ar2);
save_json(inputs_ar2, "tests/ar2-in.json");

out_ar2 = dlmfit(y_ar2, s_ar2, w_ar2, [], [], [], options_ar2);
% Also save the true states for reference
out_ar2.x_true = x_true_a2;
save_json(out_ar2, "tests/ar2-out-m.json");
disp('  done');

disp('All gensys test references generated.');
