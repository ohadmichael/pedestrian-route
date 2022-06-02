close all;

%% Allan Variance
sensor_name = 'Accelerometer';
data = readtable(sprintf('Ohad-2021-12-04_16-15-48\\%s.csv', sensor_name));
pts = 1000;
m = unique(floor(logspace(0,log10(length(data.x)/2),pts)))';


fs = 100;
[avar, t] = allanvar([data.x, data.y, data.z], m, fs);
adev = sqrt(avar);
figure;loglog(t, adev(:, 1), 'r');
hold on; loglog(t, adev(:, 2), 'g');
loglog(t, adev(:, 3), 'b');
grid on;

%% Angle Random Walk / Velocity Random Walk

b = find_slope_b(t, 1, -0.5, adev);
% Determine the angle random walk coefficient from the line.
logN = b;
N = 10.^logN;

% % Plot the results.
loglog(t, N(1) ./ sqrt(t), 'r--', t, N(2) ./ sqrt(t), 'g--', t, N(3) ./ sqrt(t), 'b--');

if strcmp(sensor_name,'Gyroscope')
    fprintf("Random walk N: (%f, %f, %f) rad/sqrt(sec) \n", N);
else
    fprintf("Random walk N: (%f, %f, %f) m/sec^1.5 \n", N);
end

%% Bias Instability
b = find_slope_b(t, 35, 0, adev);

% Determine the bias instability coefficient from the line.
scfB = sqrt(2*log(2)/pi);
logB = b - log10(scfB);
B = 10.^logB;

% Plot the results.
loglog(t, B(1) * scfB * ones(size(t)), 'r:', 'linewidth', 1.1);
loglog(t, B(2) * scfB * ones(size(t)), 'g:', 'linewidth', 1.1);
loglog(t, B(3) * scfB * ones(size(t)), 'b:', 'linewidth', 1.1);

if strcmp(sensor_name,'Gyroscope')
    fprintf("Bias instability B: (%f, %f, %f) rad/sec \n", B);
else
    fprintf("Bias instability B: (%f, %f, %f) m/sec^2 \n", B);
end

%% wrap the plot

title(sensor_name + " Allan Deviation");
xlabel("\tau [sec]");
if strcmp(sensor_name,'Gyroscope')
    ylabel("\sigma(\tau) [rad / sec^{0.5}]");
else
    ylabel("\sigma(\tau) [m / sec^{1.5}]");
end
legend('x', 'y', 'z', 'x-WN', 'y-WN', 'z-WN', 'x-BI', 'y-BI', 'z-BI');
hold off;

%% helper function
function b = find_slope_b(T, T_stop, slope, adev)
    % Find the index where the slope of the log-scaled Allan deviation is equal
    % to the slope specified.
    logtau = log10(T);
    logadev = log10(adev);
    [~, T_stop_i] = min(abs(T_stop-T));
    dlogadev = diff(logadev(1:T_stop_i, :), 1, 1) ./ diff(logtau(1:T_stop_i));
    [~, slope_i] = min(abs(dlogadev - slope), [], 1);
    
    % Find the y-intercept of the line.
    b = [logadev(slope_i(1), 1) - slope*logtau(slope_i(1)), logadev(slope_i(2), 2) - slope*logtau(slope_i(2)), logadev(slope_i(3), 3) - slope*logtau(slope_i(3))];
end