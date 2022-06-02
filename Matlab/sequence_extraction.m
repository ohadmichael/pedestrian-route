function sequence_extraction(data_directory, show_plots) 

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

% accel_data.x(1:1) = 0; accel_data.x(5000:end) = 0;
% accel_data.y(1:1) = 0; accel_data.y(5000:end) = 0;
% accel_data.z(1:1) = 0; accel_data.z(5000:end) = 0;
% gyro_data.x(1:1) = 0; gyro_data.x(5000:end) = 0;
% gyro_data.y(1:1) = 0; gyro_data.y(5000:end) = 0;
% gyro_data.z(1:1) = 0; gyro_data.z(5000:end) = 0;

% constants
fs = 100;
t = 0:1/100:(length(accel_data.time)-1)/100;
f = fs*linspace(-1/2,1/2-1/length(accel_data.time),length(accel_data.time));

% apply LPF on x, y, z axes
accel_filt_y = apply_adapted_LPF(accel_data.y, f, fs);
accel_filt_x = apply_adapted_LPF(accel_data.x, f, fs);
accel_filt_z = apply_adapted_LPF(accel_data.z, f, fs);

gyro_filt_y = apply_adapted_LPF(gyro_data.y, f, fs);
gyro_filt_x = apply_adapted_LPF(gyro_data.x, f, fs);
gyro_filt_z = apply_adapted_LPF(gyro_data.z, f, fs);

% calc norm of the data and plot
all_axes_norm = vecnorm([accel_filt_x, accel_filt_y , accel_filt_z, ...
    gyro_filt_x, gyro_filt_y, gyro_filt_z], 2, 2);

% apply summing filter (integral) on the norm and calc the energy
average_energy = conv(all_axes_norm, ones(50,1), 'same');

% cluster the energy to silence or activity, with k-means. apply median on 
% median of the energy in order to smooth the clusters
[ener_clustered, cluster_centers] = kmeans(average_energy, 2, 'Replicates', 3);
ener_clustered = medfilt1(ener_clustered, 80);

% find the indices of the activity
[~, activity_index] = max(cluster_centers);
activity_mask = ener_clustered == activity_index;
activity_rises_and_falls = diff([0; activity_mask]);
rise_indices = find(activity_rises_and_falls == 1);
fall_indices = find(activity_rises_and_falls == -1);

if (length(fall_indices) < length(rise_indices))
    fall_indices = [fall_indices; length(activity_mask)];
end

% filter low activity times
activity_times = fall_indices - rise_indices;
% need to calibrate this threshold
min_activity_time = 200;
rise_indices = rise_indices(activity_times > min_activity_time);
fall_indices = fall_indices(activity_times > min_activity_time);

if (show_plots)
    figure;
    plot(t, average_energy, "magenta", 'LineWidth', 1.5);
    hold on;
    %plot(t, all_axes_norm, "cyan");
    plot(t, 10*ener_clustered, 'LineWidth', 1.5);
    %stem(t(rise_indices), 5*ones(1, length(rise_indices)));
    %stem(t(fall_indices), -5*ones(1, length(fall_indices)));
    xlabel('time [sec]');
    ylabel('energy');
    title('Activity Clustering');
    hold off;
end

writematrix([rise_indices, fall_indices], 'sequences.csv');
end
