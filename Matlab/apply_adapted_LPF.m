function [filtered, cut_freq] = apply_adapted_LPF(axis_data, f, fs)
    fft_db = mag2db(abs(fftshift(fft(axis_data))));
    cut_freq = f(find(fft_db>=max(fft_db)-3, 1));
    filtered = lowpass(axis_data, abs(cut_freq)+1e-10, fs);
end
