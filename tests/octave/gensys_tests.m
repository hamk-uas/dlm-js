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

disp('All gensys test references generated.');
