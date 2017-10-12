%% Generate ESN
params.input_dimension = 1;
params.reservoir_dimension = 400;
params.output_dimension = 1;
params.connectivity = 0.05;
params.spectral_radius = 0.8 * 0.85;
params.output_feedback_scale = 1.0;

esn = esn_setup(params);

%% Train
ESN_TYPE = 'ESN_LEAKY';	%or 'ESN_LEAKY'
options.initial_run_length = 1000;
options.train_run_length = 2000;
options.free_run_length = 0;
options.test_run_length = 200;

mgs = generate_mgs(10000);
train_x = ones(1, 10000) * 0.2;

if strcmp(ESN_TYPE, 'ESN') == 1
	esn = esn_learn(esn, train_x, mgs, options);
elseif strcmp(ESN_TYPE, 'ESN_LEAKY') == 1
	options.leaky_learning_rate = 0.99;
	esn = esn_learn_leaky(esn, train_x, mgs, options);
end

%% Test
opts.initial_run_length = 2000;
opts.test_run_length = 84;
mgs2 = generate_mgs(10000);
test_x = ones(1, 10000) * 0.2;
if strcmp(ESN_TYPE, 'ESN') == 1
	esn_test(esn, test_x, mgs2, opts);
elseif strcmp(ESN_TYPE, 'ESN_LEAKY') == 1
	opts.leaky_learning_rate = 0.99;
	esn_test_leaky(esn, test_x, mgs2, opts);
end


