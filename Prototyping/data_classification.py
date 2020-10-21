import data_methods as dm
import numpy as np
import os as os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


file_path = 'augmented_data'
aim_path = 'regressed_data'
file_name = ''
file_paths = []

for file in os.listdir(file_path):
    if file.endswith((".csv")) and file.find(file_name) != -1:
        # takes data from specified path with specified name
        file_paths.append(os.path.join(file_path, file))

# reads out data and labels
X = []
y = []
file_names = []
for file in file_paths:
    read_data = dm.read_file(file)
    values = read_data[:-3]
    label = read_data[-3:]  #only takes angel first, might take refr error later too
    if label[2] != 0:
        X.append(values)
        y.append(dm.calcangle(label[0], label[1]))
        file_names.append(file)

# splits into train and test set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=42)

#trains the Classifier using Sklearn
"""
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf._fit(X_train, y_train)

y_predict = clf.predict(X_test)
"""

# calculates and prints the losses
losses = []
for i in range(len(y_test)):
    loss = np.sqrt(dm.quad_single_loss(dm.calclabel(y_predict[i]), dm.calclabel(y_test[i])))
    print('file: ', file_names[i], '\n Loss = ', loss)
    losses.append(loss)

# calculated mean, var and plotts everything
mean = np.mean(losses)
var = np.var(losses)
print('Mean Loss = ',mean , '   Var Loss', var)

plt.hist(losses, bins=20)
plt.xlabel('Loss')
plt.ylabel('Number of Samples')
plt.title('Mean: ' + str(mean) + '   Var: ' + str(var))
plt.show()

