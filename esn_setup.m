function esn = esn_setup(params)

reservoir_dimension = params.reservoir_dimension;
input_dimension = params.input_dimension;
output_dimension = params.output_dimension;

connectivity = params.connectivity;
spectral_radius = params.spectral_radius;
output_feedback_scale = params.output_feedback_scale;

reservoir_weights = sprand(reservoir_dimension, reservoir_dimension, connectivity);
reservoir_weights = spfun(@minusPoint5,reservoir_weights);
reservoir_weights = reservoir_weights * spectral_radius;

input_weights = 2.0 * rand(reservoir_dimension, input_dimension) - 1.0;

output_weights = zeros(output_dimension, input_dimension + reservoir_dimension);

feedback_weights = output_feedback_scale * (2.0 * rand(reservoir_dimension, output_dimension) - 1.0);

esn.input_weights = input_weights;
esn.reservoir_weights = reservoir_weights;
esn.output_weights = output_weights;
esn.feedback_weights = feedback_weights;

end