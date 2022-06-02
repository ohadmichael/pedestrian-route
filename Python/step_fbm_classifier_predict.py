import numpy as np
import pandas as pd
from joblib import load

col_count = 260

input_path_fb = '../Matlab/step_data_fb.csv'
# input_path_m = '../Matlab/step_data_m.csv'
model_path = 'step_fbm_rf_model-985-1905.joblib'

predict_data_fb = np.loadtxt(open(input_path_fb, "rb"), delimiter=",", skiprows=0, usecols=range(2, col_count - 1))
step_indices_fb = np.loadtxt(open(input_path_fb, "rb"), delimiter=",", skiprows=0, usecols=0, dtype='int')
step_seq_indices_fb = np.loadtxt(open(input_path_fb, "rb"), delimiter=",", skiprows=0, usecols=1, dtype='int')

predict_data = predict_data_fb
step_indices = step_indices_fb
step_seq_indices = step_seq_indices_fb

print('Rows in prediction set: %d' % len(predict_data))

classifier = load(model_path)

predict_labels = classifier.predict(predict_data)

predict_mat = pd.DataFrame(np.hstack((np.reshape(step_indices, (-1, 1)), np.reshape(step_seq_indices, (-1, 1)), np.reshape(predict_labels, (-1, 1)))), 
                           columns = ['step_i','seq_i','label'])
                
predict_label_count = predict_mat.groupby(['seq_i', 'label']).count().reset_index(drop=False).rename(columns={'step_i': 'steps_count'})
seq_max_votes = predict_label_count.sort_values(['seq_i', 'steps_count'], ascending=False).drop_duplicates(subset='seq_i', keep='first')

for i, row in seq_max_votes.iterrows():
    predict_mat.loc[predict_mat['seq_i'] == row['seq_i'], 'label'] = row['label']
    
predict_mat[['step_i', 'label']].to_csv('step_fbm_predictions.csv', index=False, header=False)
