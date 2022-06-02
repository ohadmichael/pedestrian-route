import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from joblib import dump
from sklearn.model_selection import train_test_split
import pandas as pd

col_count = 13

step_type = 'r'

def load_data(directory):
    data = np.zeros((0, col_count - 2))
    labels = []
    csv_files = list(filter(lambda f: f.endswith('.csv'), os.listdir(directory)))

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        file_data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0, usecols=range(1, col_count - 1))
        file_labels = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0, usecols=col_count - 1)

        data = np.concatenate((data, file_data))
        labels += list(file_labels)

    return data, labels

data, labels = load_data('../Tagged Data/step-right-length')

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

reg = LinearRegression()
reg.fit(train_data, train_labels)

print('Rows in training set: %d, testing set: %d' % (len(train_labels), len(test_labels)))
print('Chosen coefs=%s, intercept=%f' % (reg.coef_, reg.intercept_))
print('R^2: %.3f' % reg.score(test_data, test_labels))

dump(reg, 'step_%s_len_model.joblib' % step_type)

important_feat = sorted(enumerate(permutation_importance(reg, train_data, train_labels).importances_mean),
                        key=lambda t: t[1], reverse=True)[:5]

print('Most important features by permutation_importance: %s' % important_feat)

importance_coef = sorted(enumerate(np.abs(np.multiply(np.mean(train_data, axis=0), reg.coef_))),
                         key=lambda t: t[1], reverse=True)
print('Most important features by mean*coef [m]: %s' % importance_coef)

reg_test_predictions = reg.predict(test_data)
test_comp_mat = np.hstack((np.reshape(reg_test_predictions, (-1,1)), np.reshape(test_labels, (-1,1))))
diff_vec = np.abs(np.subtract(test_comp_mat[:, 0], test_comp_mat[:, 1]))
error_table = pd.DataFrame(np.hstack((test_comp_mat, np.reshape(diff_vec, (-1,1)))), columns=['est_step_size', 'real_step_size', 'diff'])
grouped_error = error_table.groupby('real_step_size').mean().reset_index()
grouped_error = grouped_error[['real_step_size','diff']]
grouped_error = grouped_error.rename(columns={'diff': 'mean_diff'})

print('Mean error per step size:')
print(grouped_error.values)

with open('train_stats.txt', 'w') as f:
    f.writelines(['Rows in training set: %d, testing set: %d\n' % (len(train_labels), len(test_labels)), 
                  'Chosen coefs=%s, intercept=%f\n' % (reg.coef_, reg.intercept_), 'R^2: %.3f\n' % reg.score(test_data, test_labels),
                  'Most important features by permutation_importance: %s\n' % important_feat, 'Most important features by mean*coef [m]: %s\n' % importance_coef,
                  'Mean error per step size: %s \n' % str(grouped_error.values)])