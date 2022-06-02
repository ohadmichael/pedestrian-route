close all; clear;

height = 185;
force_step_type = [];  % "forward/backwards" / []
check_barometer = false;
data_directory = '';

% extract sequences and classify the movement type of each sequence
sequence_extraction(data_directory, true);

if isempty(force_step_type)
    python_command = sprintf('movement_type_classifier.py %s', data_directory);
    sequences_movement_types = run_python_script(python_command, 'movement_type.csv');
else
    seq_mat = readmatrix('sequences.csv');
    sequences_movement_types = [array2table(seq_mat), table(repmat(force_step_type, size(seq_mat, 1), 1), 'VariableNames', {'stype'})];
end

% accroding to the movement type detect the steps
fb_movement = 'forward/backwards';
fb_seq_indices = sequences_movement_types{string(sequences_movement_types{:,3}) == fb_movement,1:2};
left_side_seq_indices = sequences_movement_types{string(sequences_movement_types{:,3}) == 'left',1:2};
right_side_seq_indices = sequences_movement_types{string(sequences_movement_types{:,3}) == 'right',1:2};

if iscell(fb_seq_indices), fb_seq_indices=str2double(fb_seq_indices); end
if iscell(left_side_seq_indices), left_side_seq_indices=str2double(left_side_seq_indices); end
if iscell(right_side_seq_indices), right_side_seq_indices=str2double(right_side_seq_indices); end

fb_step_indices = fb_step_detection(data_directory, fb_seq_indices(:,1), fb_seq_indices(:,2), true, true);
m_step_indices = march_step_detection(data_directory, fb_seq_indices(:,1), fb_seq_indices(:,2), fb_step_indices, true, true);
left_side_step_indices = side_step_detection(data_directory, left_side_seq_indices(:,1), left_side_seq_indices(:,2), true, true, 'l');
right_side_step_indices = side_step_detection(data_directory, right_side_seq_indices(:,1), right_side_seq_indices(:,2), true, true, 'r');

f_step_indices = [];
b_step_indices = [];
stair_step_indices = [];
step_f_lengths = cell(0,2);
step_b_lengths = cell(0,2);
step_l_lengths = cell(0,2);
step_r_lengths = cell(0,2);

if ~isempty(fb_step_indices)
    classified_steps_fb = run_python_script('step_fbm_classifier_predict.py', 'step_fbm_predictions.csv');
    f_step_indices = classified_steps_fb{strcmp(classified_steps_fb{:,2},'f'), 1};
    b_step_indices = classified_steps_fb{strcmp(classified_steps_fb{:,2},'b'), 1};
    m_step_indices = [m_step_indices; classified_steps_fb{strcmp(classified_steps_fb{:,2},'m'), 1}];

    prepare_step_len_data(height, 'fb', f_step_indices, 'f');
    prepare_step_len_data(height, 'fb', b_step_indices, 'b');
    
    if ~isempty(f_step_indices)
        step_f_lengths = run_python_script('step_len_est_predict.py f', 'step_f_len_predictions.csv');

        any_stair_cand = check_barometer && get_stairs_candidates(data_directory, fb_seq_indices, f_step_indices);
        if any_stair_cand
            python_command = sprintf('stairs_detection.py %s', data_directory);
            stairs_seq = run_python_script(python_command, 'stairs_seq.csv');
            if size(stairs_seq, 1) == 1, stairs_seq=str2double(stairs_seq{logical(str2double(stairs_seq{:, 3})), 1:2}); else, stairs_seq=stairs_seq{logical(stairs_seq{:, 3}), 1:2}; end
          
            for stair_seq_i = 1:size(stairs_seq, 1)
                stair_step_indices = [stair_step_indices; f_step_indices(f_step_indices >= stairs_seq(stair_seq_i, 1) & f_step_indices <= stairs_seq(stair_seq_i, 2))];
            end
        end
    end
    if ~isempty(b_step_indices)
        step_b_lengths = run_python_script('step_len_est_predict.py b', 'step_b_len_predictions.csv');
    end
end

if ~isempty(left_side_step_indices)
    prepare_step_len_data(height, 'l', false, '');
    step_l_lengths = run_python_script('step_len_est_predict.py l', 'step_l_len_predictions.csv');
end

if ~isempty(right_side_step_indices)
    prepare_step_len_data(height, 'r', false, '');
    step_r_lengths = run_python_script('step_len_est_predict.py r', 'step_r_len_predictions.csv');
end

classified_steps_i = [f_step_indices; b_step_indices; m_step_indices; left_side_step_indices; right_side_step_indices];
classified_steps_label = [repmat('f', length(f_step_indices), 1); repmat('b', length(b_step_indices), 1); repmat('m', length(m_step_indices), 1); repmat('l', length(left_side_step_indices), 1); repmat('r', length(right_side_step_indices), 1)];
classified_steps_len = [step_f_lengths{:, 2}; step_b_lengths{:, 2}; zeros(length(m_step_indices), 1); step_l_lengths{:, 2}; step_r_lengths{:, 2}];

[classified_steps_i, sorted_steps_ii] = sort(classified_steps_i);
classified_steps_label = classified_steps_label(sorted_steps_ii);
classified_steps_len = classified_steps_len(sorted_steps_ii);

angle_vec = get_directions(classified_steps_i, data_directory);

locations = [[0, 0]; zeros(length(classified_steps_i), 2)];
for step_i=1:length(classified_steps_i)
    step_length = classified_steps_len(step_i);
    angle = angle_vec(step_i);

    if any(strcmp(classified_steps_label(step_i), {'r', 'l'}))
        angle = angle + pi/2;
    end

    n_step = step_length*cos(angle);
    e_step = step_length*sin(angle);

    if any(strcmp(classified_steps_label(step_i), {'b', 'l'}))
        n_step = -n_step;
        e_step = -e_step;
    end

    locations(step_i+1,:) = locations(step_i,:) + [e_step, n_step];
end

figure;
plot(rad2deg(angle_vec));
title('Walking direction');

figure;

plot(locations(:,1), locations(:,2), 'DisplayName', 'Our system', 'LineWidth', 1.5);
hold on;
arrow_part1 = locations(4:10:end, :);
arrow_part2 = locations(5:10:end, :);
arrow_part1 = arrow_part1(1:size(arrow_part2, 1), :);
arrows = arrow_part2-arrow_part1;
quiver(arrow_part1(:, 1), arrow_part1(:, 2), arrows(:,1), arrows(:,2), 0.15, 'DisplayName', 'Our system - direction', 'LineWidth', 2);

gps_table = readtable(strcat(data_directory, '/Location.csv'));
lla = [gps_table.latitude, gps_table.longitude, gps_table.altitude];
gps_coords = lla2enu(lla(10:end, :), lla(10, :), 'flat');
plot(gps_coords(:, 1), gps_coords(:, 2), 'DisplayName', 'GPS', 'LineWidth', 1.5);

% scatter the step types
step_type_color = [
    ["f"; "b"; "m"; "r"; "l"], ...
    ["k"; "b"; "y"; "r"; "m"]
];
for step_type_i = 1:size(step_type_color, 1)
    step_type = step_type_color(step_type_i, 1);
    color = step_type_color(step_type_i, 2);
    step_locs = locations([false; strcmp(classified_steps_label, step_type)], :);
    step_locs_skipped = step_locs(3:10:end, :);
    scatter(step_locs_skipped(:, 1), step_locs_skipped(:, 2), 70, 'p', 'MarkerEdgeColor', color, 'DisplayName', ['Step type - ' char(step_type)]);

    if strcmp(step_type, "f")
        f_step_i = classified_steps_i(strcmp(classified_steps_label, step_type));
        stair_locs = step_locs(ismember(f_step_i, stair_step_indices), :);
        stair_locs_skipped = stair_locs(5:10:end, :);
        scatter(stair_locs_skipped(:, 1), stair_locs_skipped(:, 2), 70, '^', 'MarkerEdgeColor', color, 'DisplayName', 'Staircase climbing');
    end
end

hold off;
legend();
title('Walking map');
grid on;
max_coord = max(max(abs(locations))) + 1;
xlim([-max_coord, max_coord]);
ylim([-max_coord, max_coord]);
xlabel('E [m]');
ylabel('N [m]');

fprintf('Total distance: %.2f \n', sum(classified_steps_len));
gps_lengths = sqrt((gps_coords(2:end, 1) - gps_coords(1:end-1, 1)).^2 + (gps_coords(2:end, 2) - gps_coords(1:end-1, 2)).^2);
fprintf('GPS total distance: %.2f \n', sum(gps_lengths));
fprintf('Distance from start to end: %.2f \n', abs(locations(end,1) + 1j*locations(end,2)));
fprintf('GPS distance from start to end: %.2f \n', abs(gps_coords(end,1) + 1j*gps_coords(end,2)));
fprintf('Step count: %d \n', length(classified_steps_i));
