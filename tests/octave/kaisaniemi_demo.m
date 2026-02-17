addpath('tests/octave', 'tests/octave/dlm');

%% Kaisaniemi monthly temperature seasonal demo
% Source model setup from dlmtut.org seasonal example:
% load kaisaniemi.mat; options.order=1; options.trig=1;
% w0 = [0, 0.005, 0.4, 0.4]; s = 2;

load('tests/octave/kaisaniemi.mat'); % variables: time, temp

options = struct('order', 1, 'trig', 1);
s = 2;
w = [0, 0.005, 0.4, 0.4];

inputs = struct(
  'y', temp,
  't', time,
  's', s,
  'w', w,
  'options', options
);
save_json(inputs, 'tests/kaisaniemi-in.json');

out = dlmfit(temp, s, w, [], [], [], options);
save_json(out, 'tests/kaisaniemi-out-m.json');

disp('Kaisaniemi demo references generated.');
