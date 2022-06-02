function [any_stair_cand] = get_stairs_candidates(data_directory, seq_indices, step_indices)
    fs_bar = 1;
    fs_accel = 100;
    mbar_in_meters = 8.323;
    slope_threshold = 0.1;
    bar_data = readtable(strcat(data_directory, '/Barometer.csv'));
    altitude = (bar_data.pressure-bar_data.pressure(1))*(-mbar_in_meters);
    seq_is_stair_cand = false(size(seq_indices, 1), 1);

    fb_bar_seq_indices = floor(seq_indices * (fs_bar/fs_accel));
    
    for i = 1:size(fb_bar_seq_indices, 1)
        if(any(step_indices >= seq_indices(i, 1) & step_indices <= seq_indices(i, 2)))
            alt_seq_start = altitude(fb_bar_seq_indices(i,1));
            alt_seq_end = altitude(fb_bar_seq_indices(i,2));
            avg_slope = (alt_seq_end-alt_seq_start)/(fb_bar_seq_indices(i,2)-fb_bar_seq_indices(i,1));
            seq_is_stair_cand(i) = avg_slope > slope_threshold;
        end
    end

    any_stair_cand = any(seq_is_stair_cand);

    if any_stair_cand
        export_mat = seq_indices(seq_is_stair_cand, :);
        writematrix(export_mat, 'stair_cand_indices.csv');
    end
end
