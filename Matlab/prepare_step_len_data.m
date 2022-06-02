function prepare_step_len_data(height, step_type, required_step_indices, output_step_type)
    data = readmatrix(['/step_data_' step_type '.csv']);
    if isempty(data)
        writematrix([], ['step_' step_type '_length_data.csv']);
        return;
    end

    if ~islogical(required_step_indices)
        data = data(ismember(data(:, 1), required_step_indices), :);
    end

    if isempty(output_step_type)
        output_step_type = step_type;
    end

    step_indices = data(:, 1);
    z_accel = data(:, 3:102);
    z_gyro = data(:, 153:252);
    sample_count = sum(z_accel~=0, 2);
    accel_z_energy_sq = sqrt(sum(z_accel.^2, 2));
    gyro_z_energy_sq = sqrt(sum(z_gyro.^2, 2));
    [accel_z_max, accel_z_max_i] = max(z_accel, [], 2);
    [accel_z_min, accel_z_min_i] = min(z_accel, [], 2);
    ptp_dist = abs(accel_z_max_i-accel_z_min_i);
    accel_z_std = std(z_accel, 0, 2);

    export_mat = [
        step_indices, ... # step start index
        sample_count, ... # z-accelerometer sample count (time)
        1./sample_count, ... # z-accelerometer inverse sample count (frequency)
        sample_count.^2, ... # z-accelerometer squared sample count
        accel_z_energy_sq, ... # z-accelerometer sqrt energy
        gyro_z_energy_sq, ... # z-gyro sqrt energy
        accel_z_max, ... # z-accelerometer max
        accel_z_min, ... # z-accelerometer min  
        ptp_dist, ... # z-accelerometer distance between max and min  
        accel_z_std, ... # z-accelerometer std
        accel_z_std.^2, ... # z-accelerometer variance
        repmat(height, length(step_indices), 1)
    ];
    writematrix(export_mat, ['step_' output_step_type '_length_data.csv']);
end