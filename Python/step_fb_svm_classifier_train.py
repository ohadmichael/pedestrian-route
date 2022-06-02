#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from joblib import dump
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split

col_count = 260


def load_data(directory):
    data = np.zeros((0, col_count - 3))
    labels = []
    csv_files = list(filter(lambda f: f.endswith('.csv'), os.listdir(directory)))

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        file_data = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0, usecols=range(2, col_count - 1))
        file_labels = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0, usecols=col_count - 1, dtype='str')

        data = np.concatenate((data, file_data))
        labels += list(file_labels)

    return data, labels

data, labels = load_data(r"C:\Users\Ohad\Documents\Final project\Python\step_train\tagged new 13-05 with 100 gyro samples")
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(train_data, train_labels)

print('Rows in training set: %d, testing set: %d' % (len(train_labels), len(test_labels)))
print('Chosen kernel=%s, C=%d' % (svc.kernel, svc.C))

print('Accuracy: %.3f' % clf.score(test_data, test_labels))

dump(clf, 'step_fb_svm_model.joblib')

important_feat = sorted(enumerate(permutation_importance(clf, train_data, train_labels).importances_mean),
                        key=lambda t: t[1], reverse=True)[:5]

print('Most important features: %s' % important_feat)

plot_confusion_matrix(clf, test_data, test_labels)
plt.savefig('step_fb_conf_mat.png')
plt.show()

with open('train_stats.txt', 'w') as f:
    f.writelines(['Rows in training set: %d, testing set: %d' % (len(train_labels), len(test_labels)), 
                  'Chosen kernel=%s, C=%d' % (svc.kernel, svc.C), 'Accuracy: %.3f' % clf.score(test_data, test_labels),
                  'Most important features: %s' % important_feat])
