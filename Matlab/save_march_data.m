close all;

force_step_type = "forward/backwards";  % "" / []
data_directory = '../tamir_20_marches-2022-03-09_16-47-43';

seq_indx = [1];

% extract sequences and classify the movement type of each sequence
sequence_extraction(data_directory, true);

seq_mat = readmatrix('sequences.csv');
seq_mat = seq_mat(seq_indx, :);
sequences_movement_types = [array2table(seq_mat), table(repmat(force_step_type, size(seq_mat, 1), 1), 'VariableNames', {'stype'})];

fb_seq_indices = sequences_movement_types{string(sequences_movement_types{:,3}) == fb_movement,1:2};

march_step_indices = march_step_detection(data_directory, fb_seq_indices(:,1), fb_seq_indices(:,2), [],true, true);
