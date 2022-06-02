function step_angle_vec = get_directions(steps_indices, data_directory)
    magneto_data = readtable(strcat(data_directory, '/Magnetometer.csv'));
    fs = 100;
    f = fs*linspace(-1/2,1/2-1/length(magneto_data.time),length(magneto_data.time));

    % apply filters on the relevant axes
    magneto_x_filtered = apply_adapted_LPF(magneto_data.x, f, fs);
    magneto_x_med_fil = medfilt1(magneto_x_filtered, 100);
    
    magneto_z_filtered = apply_adapted_LPF(magneto_data.z, f, fs);
    magneto_z_med_fil = medfilt1(magneto_z_filtered, 100);
    
    % find angle out of x,z axes
    x_z_vec = -magneto_z_med_fil - 1j.*magneto_x_med_fil;
    angle_vec = angle(x_z_vec);
    
    med_filt_length = 9;
    angle_vec_med = medfilt1(angle_vec, med_filt_length);
    step_angle_vec = angle_vec_med(steps_indices + floor(med_filt_length/2)) + deg2rad(30);
end
