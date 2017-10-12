function esn = esn_learn(esn, train_x, train_y, options)

input_dimension = size(esn.input_weights, 2);
reservoir_dimension = size(esn.reservoir_weights, 1);
output_dimension = size(esn.output_weights, 1);
total_dimension = input_dimension + reservoir_dimension + output_dimension;

initial_run_length = options.initial_run_length;
train_run_length = options.train_run_length;
free_run_length = options.free_run_length;
test_run_length = options.test_run_length;
leaky_learning_rate = options.leaky_learning_rate;

noise_level = 0.0000000001;

total_state = zeros(total_dimension, 1);
state_collect_matrix = zeros(train_run_length, reservoir_dimension + input_dimension);
target_collect_matrix = zeros(train_run_length, output_dimension);

%% In A Single Loop
mse_test = zeros(1, output_dimension); 
mse_train = zeros(1, output_dimension); 
reservoir_state = 0;

for i = 1:initial_run_length + train_run_length + free_run_length + test_run_length 
    
    input = train_x(:,i);  
    target_output = train_y(:,i);    
    
    total_state(reservoir_dimension+1:reservoir_dimension+input_dimension) = input; 

    if i > initial_run_length + train_run_length
        reservoir_state = leaky_learning_rate * f([esn.reservoir_weights, esn.input_weights, esn.feedback_weights]*total_state) + ...
			(1-leaky_learning_rate) * reservoir_state;
    else
        reservoir_state = leaky_learning_rate * f([esn.reservoir_weights, esn.input_weights, esn.feedback_weights]*total_state + ...
			(1-leaky_learning_rate) * reservoir_state + ...
            noise_level * 2.0 * (rand(reservoir_dimension,1)-0.5));
    end
    
    pratical_output = f(esn.output_weights * [reservoir_state;input]);
    total_state = [reservoir_state; input; pratical_output];    
    
    if (i > initial_run_length) && (i <= initial_run_length + train_run_length) 
        collectIndex = i - initial_run_length;
        state_collect_matrix(collectIndex,:) = [reservoir_state' input']; 
        target_collect_matrix(collectIndex,:) = (fInverse(target_output))';
    end
    
    if i <= initial_run_length + train_run_length
        total_state(reservoir_dimension+input_dimension+1:reservoir_dimension+input_dimension+output_dimension) = target_output' ; 
    end
    
    if i > initial_run_length + train_run_length + free_run_length
        for j = 1:output_dimension
            mse_test(1,j) = mse_test(1,j) + (target_output(j,1)- pratical_output(j,1))^2;
        end
    end

    if i == initial_run_length + train_run_length
        esn.output_weights = (pinv(state_collect_matrix) * target_collect_matrix)'; 
        for j = 1:output_dimension
            mse_train(1,j) = sum((f(target_collect_matrix(:,j)) - ...
                    f(state_collect_matrix * esn.output_weights(j,:)')).^2);
            mse_train(1,j) = mse_train(1,j) / train_run_length;
        end
    end  
end

mse_test = mse_test / test_run_length;

disp(sprintf('train MSE = %g   test MSE = %g   avWeights = %g', ...
    mse_train , mse_test, mean(abs(esn.output_weights))));
end