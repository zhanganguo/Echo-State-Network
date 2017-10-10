function esn_test(esn, test_x, test_y, options)

input_dimension = size(esn.input_weights, 2);
reservoir_dimension = size(esn.reservoir_weights, 1);
output_dimension = size(esn.output_weights, 1);
total_dimension = input_dimension + reservoir_dimension + output_dimension;

initial_run_length = options.initial_run_length;
test_run_length = options.test_run_length;

total_state =  zeros(total_dimension,1);    

for i = 1:initial_run_length   
    input = test_x(:, i);
    target_output = test_y(:, i);   
    total_state(reservoir_dimension+1:reservoir_dimension+input_dimension) = input; 
    reservoir_state = f([esn.reservoir_weights, esn.input_weights, esn.feedback_weights]*total_state); 
    pratical_output = f(esn.output_weights *[reservoir_state; input]);
    total_state = [reservoir_state;input; pratical_output];      
    total_state(reservoir_dimension+input_dimension+1: reservoir_dimension+input_dimension+output_dimension) = target_output' ; 
end
startstate = total_state;

numberOfTrials = 50;
output_collect_matrix = zeros(test_run_length, numberOfTrials);
teacher_collect_matrix = zeros(test_run_length, numberOfTrials);
trialshift = 84;
for trials = 1:numberOfTrials
    total_state = startstate;
    for i = 1:trialshift   
        index = initial_run_length + i + (trials - 1)*trialshift;
        input = test_x(:, index);     
        target_output = test_y(:, index);         
        
        total_state(reservoir_dimension+1:reservoir_dimension+input_dimension) = input; 
        
        reservoir_state = f([esn.reservoir_weights, esn.input_weights, esn.feedback_weights]*total_state); 
        pratical_output = f(esn.output_weights *[reservoir_state;input]);
        total_state = [reservoir_state;input;pratical_output];      
        total_state(reservoir_dimension+input_dimension+1:reservoir_dimension+input_dimension+output_dimension) = target_output' ; 
    end
    startstate = total_state;
    
    for i = 1:test_run_length  
        index = initial_run_length + i + (trials - 1)*trialshift + trialshift;
        input = test_x(:, index);     
        target_output = test_y(:, index);         
        
        total_state(reservoir_dimension+1:reservoir_dimension+input_dimension) = input; 
          
        reservoir_state = f([esn.reservoir_weights, esn.input_weights, esn.feedback_weights]*total_state); 
        pratical_output = f(esn.output_weights *[reservoir_state;input]);
        total_state = [reservoir_state;input;pratical_output];      
        output_collect_matrix(i, trials) = pratical_output;
        teacher_collect_matrix(i, trials) = target_output;
    end
end

% compute the normalized root MSE for 84 step prediction

% compute variance of original MG series
teachVar = var(fInverse(test_y));
% undo the tanh transformation of the training data to recover (shifted)
% original MG data
outputCollectMatMG = fInverse(output_collect_matrix);
teacherCollectMatMG = fInverse(teacher_collect_matrix);
% compute average NMSE of 84 step predictions
errors84 = outputCollectMatMG(84,:) - teacherCollectMatMG(84,:);
NRMSE84 = sqrt(mean(errors84.^2)/teachVar) % result 3.0999e-005

end