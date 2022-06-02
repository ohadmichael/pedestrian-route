import numpy as np
from joblib import load
import sys

col_count = 13

step_type = sys.argv[1]
input_path = '../Matlab/step_%s_length_data.csv' % step_type
model_path = 'step_%s_len_model.joblib' % step_type

predict_data = np.loadtxt(open(input_path, "rb"), delimiter=",", skiprows=0, usecols=range(1, col_count - 1))
step_indices = np.loadtxt(open(input_path, "rb"), delimiter=",", skiprows=0, usecols=0, dtype='int')

print('Rows in prediction set (%s): %d' % (step_type, len(predict_data)))

classifier = load(model_path)

predict_labels = classifier.predict(predict_data)

predict_mat = np.hstack((np.reshape(step_indices, (-1, 1)), np.reshape(predict_labels, (-1, 1))))
np.savetxt('step_%s_len_predictions.csv' % step_type, predict_mat, delimiter=",", fmt='%s')
