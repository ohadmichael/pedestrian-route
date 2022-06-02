function [step_indices] = side_step_detection(data_directory, rise_indices, fall_indices, show_plots, export_steps, step_type)

accel_data = readtable(strcat(data_directory, '/Accelerometer.csv'));
gyro_data = readtable(strcat(data_directory, '/Gyroscope.csv'));

if (size(accel_data, 1) ~= size(gyro_data, 1))
    vec_length = min(size(accel_data, 1), size(gyro_data, 1));
    accel_data = accel_data(1:vec_length, :);
    gyro_data = gyro_data(1:vec_length, :);
end

should_flip_accel = contains(lower(data_directory), 'tamir');
if (should_flip_accel)
    accel_data.x = -accel_data.x;
    accel_data.y = -accel_data.y;
    accel_data.z = -accel_data.z;
end

% constants
fs = 100;
t = 0:1/100:(length(accel_data.time)-1)/100;
f = fs*linspace(-1/2,1/2-1/length(accel_data.time),length(accel_data.time));

% apply LPF on x, y, z axes
accel_filt_y = apply_adapted_LPF(accel_data.y, f, fs);
[accel_filt_x, accel_x_cut_freq] = apply_adapted_LPF(accel_data.x, f, fs);
accel_filt_z = apply_adapted_LPF(accel_data.z, f, fs);

gyro_filt_y = apply_adapted_LPF(gyro_data.y, f, fs);
gyro_filt_x = apply_adapted_LPF(gyro_data.x, f, fs);
gyro_filt_z = apply_adapted_LPF(gyro_data.z, f, fs);

%% find peaks in each activity time

min_peak_distance = (1/abs(accel_x_cut_freq))*fs / 2;
peak_dis_std_threshold = 0.15;
sequence_corr_max_peaks = 15;

steps = zeros(1e3, 150);
step_indices = zeros(1e3, 1);
step_lengths = zeros(1e3, 1);
step_sequence_indexes = zeros(1e3, 1);
step_i = 1;

for rise_i=1:length(rise_indices)
    sequence = accel_filt_x(rise_indices(rise_i):fall_indices(rise_i));
    [~, peak_locs] = findpeaks(sequence, "MinPeakDistance", min_peak_distance, "MinPeakHeight", 0);
    
    chunk_count = ceil(length(peak_locs)/sequence_corr_max_peaks);
    for chunk_i=1:chunk_count
        % applying correlation for detecting False Alarms
        if chunk_i==1
            start_i = 1;
        else
            prev_last_peak_i = (chunk_i-1)*sequence_corr_max_peaks;
            start_i = floor(mean(peak_locs(prev_last_peak_i:prev_last_peak_i+1)));
        end

        if chunk_i==chunk_count
            end_i = length(sequence);
        else
            last_peak_i = (chunk_i)*sequence_corr_max_peaks;
            end_i = floor(mean(peak_locs(last_peak_i:last_peak_i+1)));
        end
        
        first_peak_i = (chunk_i-1)*sequence_corr_max_peaks+1;
        if chunk_i==chunk_count && length(peak_locs) == first_peak_i
            correlator_end_i = end_i;
        else
            correlator_end_i = floor(mean(peak_locs(first_peak_i:first_peak_i+1)));
        end
        
        chunk = sequence(start_i:end_i);
        correlator = sequence(start_i:correlator_end_i);
        correlation = xcorr(correlator, chunk);

        % apply findpeaks to find std of peak distances
        [~, corr_peak_locs] = findpeaks(correlation, "MinPeakHeight", max(correlation)/10, "MinPeakProminence", max(correlation)/5);

        if length(corr_peak_locs) > 1
            peak_dis_diff = rmoutliers(diff(corr_peak_locs), 'gesd', 'MaxNumOutliers', 1);
            peak_dis_std = std(peak_dis_diff)/max(peak_dis_diff);
        else
            peak_dis_std = 0;
        end

        if peak_dis_std < peak_dis_std_threshold
            chunk_peaks = peak_locs(peak_locs > start_i & peak_locs < end_i);
            chunk_step_bounds = [start_i; chunk_peaks(1:end-1) + floor(diff(chunk_peaks)/2); end_i];

            for chunk_step_i=1:length(chunk_step_bounds)-1
                step = sequence(chunk_step_bounds(chunk_step_i):chunk_step_bounds(chunk_step_i+1));
                steps(step_i, 1:length(step)) = step;
                step_indices(step_i) = rise_indices(rise_i) + chunk_step_bounds(chunk_step_i) -1;
                step_lengths(step_i) = length(step);
                step_sequence_indexes(step_i) = rise_indices(rise_i);
                step_i = step_i + 1;
            end
        end
    end
end

if (show_plots && any(step_indices))
    % plot all axes data
    % accelerometer
    figure;
    plot(t, accel_filt_y, "g");
    hold on;
    plot(t, accel_filt_x, "r");
    plot(t, accel_filt_z, "b");
    
    stem(t(rise_indices), 5*ones(1, length(rise_indices)));
    stem(t(fall_indices), -5*ones(1, length(fall_indices)));
    
    figure;
    plot(t, accel_filt_x, 'LineWidth', 2);
    hold on;
    stem(t(step_indices(step_indices ~= 0)), 2*ones(length(find(step_indices))), 'r', 'LineWidth', 1);
    hold off;
    title('Step Extraction');
    xlabel('time [sec]');
    ylabel('acceleration x-axis [m/sec^2]');
end

%% export steps
if export_steps
    step_count = step_i-1;
    gyro_steps = zeros(step_count, 150);
    for step_i=1:step_count
        gyro_steps(step_i, 1:step_lengths(step_i)) = gyro_filt_x(step_indices(step_i):step_indices(step_i)+step_lengths(step_i)-1);
    end
    accel_x_fft = fft(steps, 50, 2);
    step_indices = step_indices(1:step_count);
    
    export_mat = [
        step_indices, ... # step start index
        steps(1:step_count, 1:100), ... # x-acceleration step samples
        abs(accel_x_fft(1:step_count, :)), ... # x-accelerometer step samples FFT
        gyro_steps(:, 1:100), ... # x-gyro step samples
        accel_filt_x(step_sequence_indexes(1:step_count)), ... # sequence start x-acceleration
        mean(abs(steps(1:step_count, :)).^2, 2), ... # x-acceleration mean power
        repmat(abs(accel_x_cut_freq), step_count, 1), ... # x-acceleration cut frequency    
        accel_filt_z(step_indices), ... # step start z-acceleration 
        accel_filt_y(step_indices), ... # step start y-acceleration 
        gyro_filt_z(step_indices), ... # step start z-gyro
        gyro_filt_y(step_indices) ... # step start y-gyro
    ];
    writematrix(export_mat, ['step_data_' step_type '.csv']);
end

end
