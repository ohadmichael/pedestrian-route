import pandas as pd
import numpy as np
import cv2 as cv
import sys
import math
import itertools
from scipy.ndimage import median_filter

data_directory = sys.argv[1]

video_filename = data_directory + r"/walking_video.mp4"
csv_sequences_filename = "../Matlab/stair_cand_indices.csv"
output_csv_filename = "stairs_seq.csv"
time_offset = 10  # in seconds
sensor_frequency = 100  # in hertz
fps = 30
processing_frames_factor = 0.3

# read the sequence csv
sequences = pd.read_csv(csv_sequences_filename, header=None)
sequences_classification = sequences.copy()
sequences_classification["stairs"] = " "
sequences = sequences / sensor_frequency
sequences = sequences - time_offset
sequences = sequences[np.all(sequences > 0, 1)]

cap = cv.VideoCapture(video_filename)
ret, frame = cap.read()

max_horiz_deviation_angle = 5
max_angle_diff_threshold = 1
min_line_dist = 10
max_line_dist = frame.shape[0] * 0.12
min_seq_frames_for_stairs = 2


def frame_has_stairs(frame):
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    filt_image = cv.bilateralFilter(gray_image, 4 * 5, 10, 5)

    edges = cv.Canny(filt_image, 100, 200)

    for filter_size in range(5, 50, 2):
        filt_edges = median_filter(edges, size=(1, filter_size))
        lines = cv.HoughLines(filt_edges, 1, np.pi / 180, 150)

        if lines is None:
            return False

        if lines.shape[0] < 10:
            print('filter size: %d' % filter_size)
            break

    print('lines found: %d' % lines.shape[0])
    deg_lines = [(line[0][0], round(line[0][1] * 180 / np.pi)) for line in lines]
    line_combs = list(itertools.combinations(deg_lines, 3))

    for line_comb in line_combs:
        loc_arr = np.array([line[0] for line in line_comb])
        angle_arr = np.array([line[1] for line in line_comb])
        if np.all(np.abs(angle_arr - 90) <= max_horiz_deviation_angle) and np.max(angle_arr) - np.min(
                angle_arr) <= max_angle_diff_threshold:
            sorted_loc_arr = np.sort(loc_arr)
            dist1 = sorted_loc_arr[1] - sorted_loc_arr[0]
            dist2 = sorted_loc_arr[2] - sorted_loc_arr[1]
            if np.all(np.greater([dist1, dist2], min_line_dist)) and np.all(np.less([dist1, dist2], max_line_dist)):
                dist_diff = abs(dist1 - dist2)
                if dist_diff < 0.1 * max_line_dist:
                    return True

    return False


for seq_i in range(sequences.shape[0]):
    # iterate through the sequnces
    # take first frame of the video
    sequence_start = max(0, sequences.iloc[seq_i, 0] - 1)
    sequence_end = sequences.iloc[seq_i, 1]

    stair_frame_counter = 0

    cap.set(1, math.ceil(fps * sequence_start))
    ret, frame = cap.read()

    sequence_frames = min(math.floor((sequence_end - sequence_start) / processing_frames_factor), 30)
    for j in range(sequence_frames):
        if frame_has_stairs(frame):
            stair_frame_counter += 1
            if stair_frame_counter == min_seq_frames_for_stairs:
                break

        cap.set(1, cap.get(1) + math.floor(processing_frames_factor * fps) - 1)
        ret, frame = cap.read()

    sequences_classification.iloc[seq_i, 2] = int(stair_frame_counter == min_seq_frames_for_stairs)

sequences_classification.to_csv(output_csv_filename, header=None, index=False)
