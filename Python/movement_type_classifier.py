# reads csv with sequences start and end index, and a video file.
# saves a csv with sequence start and end index and direction.

import pandas as pd
import numpy as np
import cv2 as cv
import math
import sys

data_directory = sys.argv[1]

video_filename = data_directory + r"/walking_video.mp4"
csv_sequences_filename = "../Matlab/sequences.csv"
output_csv_filename = "movement_type.csv"
time_offset = 10 # in seconds
sensor_frequency = 100 # in hertz
fps = 30
processing_frames_factor = 0.5

# read the sequence csv
sequences = pd.read_csv(csv_sequences_filename, header=None)
sequences_classification = sequences.copy()
sequences_classification["movement type"] = " "
sequences = sequences / sensor_frequency
sequences = sequences - time_offset

cap = cv.VideoCapture(video_filename)
ret,frame = cap.read()

# setup initial location of window
frame_size = frame.shape
x, y, w, h = 2*frame_size[1]//5, 2*frame_size[0]//5, frame_size[1]//5, frame_size[0]//5 # simply hardcoded the values
track_window = (x, y, w, h)

for seq_i in range(sequences.shape[0]):
    left_votes = []
    right_votes = []
    
    # iterate through the sequnces
    # take first frame of the video
    sequence_start = sequences.iloc[seq_i,0]
    sequence_end = sequences.iloc[seq_i,1]

    if sequence_start < 0:
        sequences_classification.iloc[seq_i, 2] = 'noise'
        continue
    
    cap.set(1, math.ceil(fps*sequence_start))
    ret,frame = cap.read()
    
    sequence_frames = min(math.floor((sequence_end - sequence_start) / processing_frames_factor), 20)
    for j in range(sequence_frames):
        first_frame_box = frame[y:y+h, x:x+w]
        first_frame_box_hue = cv.cvtColor(first_frame_box, cv.COLOR_BGR2HSV)[:,:,0]

        # get the next frame for comparison
        cap.set(1,cap.get(1)+math.floor(processing_frames_factor*fps)-1) 
        ret, frame = cap.read()
        second_frame_hue = cv.cvtColor(frame, cv.COLOR_BGR2HSV)[:,:,0]

        min_cost = np.inf
        min_cost_index = (0, 0)

        # search for the box in the new frame
        for y_cur in range(frame.shape[0]//10, round(0.9*frame.shape[0]) - h, 5):
            for x_cur in range(frame.shape[1]//10, round(0.9*frame.shape[1]) - w, 5):
                current_box = second_frame_hue[y_cur:y_cur+h, x_cur:x_cur+w]
                mse = np.mean(np.power(current_box - first_frame_box_hue, 2))
                cost = mse + ((y_cur-y)**2+(x_cur-x)**2)/1500

                # check if the cost is lower than the minimum cost so far
                if cost < min_cost:
                    min_cost = cost
                    min_cost_index = (x_cur, y_cur)         

        x_diff = min_cost_index[0] - x
        threshold = 5
        left_votes.append(x_diff > threshold)
        right_votes.append(x_diff < -threshold)

    left_percentage = sum(left_votes) / sequence_frames
    right_percentage = sum(right_votes) / sequence_frames
    
    if left_percentage > 0.6:
        sequences_classification.iloc[seq_i,2] = 'left'
    elif right_percentage > 0.6:
        sequences_classification.iloc[seq_i,2] = 'right'
    else:
        sequences_classification.iloc[seq_i,2] = 'forward/backwards'
        
sequences_classification.to_csv(output_csv_filename, header=None, index=False)
