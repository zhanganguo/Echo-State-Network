%% Generate ESN
params.input_dimension = 1;
params.reservoir_dimension = 400;
params.output_dimension = 1;
params.connectivity = 0.05;
params.spectral_radius = 0.8 * 0.85;
params.output_feedback_scale = 1.0;

esn = esn_setup(params);

%% Train
options.initial_run_length = 1000;
options.train_run_length = 2000;
options.free_run_length = 0;
options.test_run_length = 200;

mgs = generate_mgs(10000);
train_x = ones(1, 10000) * 0.2;

esn = esn_learn(esn, train_x, mgs, options);

%% Test
opts.initial_run_length = 2000;
opts.test_run_length = 84;
mgs2 = generate_mgs(10000);
test_x = ones(1, 10000) * 0.2;
esn_test(esn, test_x, mgs2, opts);

