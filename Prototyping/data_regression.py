import data_methods as dm
import numpy as np
import os as os
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder



# reads in the file and splits into data and label
file_path = 'augmented_data'
aim_path = 'regressed_data'
file_name = ''
file_paths = []

for file in os.listdir(file_path):
    if file.endswith((".csv")) and file.find(file_name) != -1:
        # takes data from specified path with specified name
        file_paths.append(os.path.join(file_path, file))


# Reads in all the data and puts it into X and Y
# file_names is for later recognition of the datapoints
X =[]
y = []
file_names = []
for file in file_paths:
    read_data = dm.read_file(file)
    values = read_data[:-3]
    label = read_data[-3:]  #only takes angel first, might take refr error later too
    if label[2] != 0:
        X.append(values, )
        y.append(label[:2])
        file_names.append(file)

# splits into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33 , random_state=42,)

#trains the Classifier using Sklearn
"""
y_train_bin = [dm.calcangle(x[0], x[1]) for x in y_train]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf._fit(X_train, y_train_bin)

y_predict_bin = clf.predict(X_test)
y_predict = [dm.calclabel(x) for x in y_train_bin]

"""

#trains the NN using sklearn
"""
clf = MLPRegressor(hidden_layer_sizes=(5,2), activation='relu', random_state=1)

clf.fit(X_train, y_train)
plt.plot(clf.loss_curve_)
plt.figure()

y_predict = clf.predict(X_test)
"""

# trains using a Decision tree and sklearn
"""
clf = tree.DecisionTreeRegressor(max_depth=10)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
"""

# trains using keras and Regression
"""
batch_size = 235
num_of_output_dimensions = 2
epochs = 500

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(450,)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_of_output_dimensions, activation='elu'))

model.summary()

loss_obj = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.AUTO)
opt = keras.optimizers.RMSprop(learning_rate=0.002)
model.compile(loss=loss_obj,
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(X_test), np.array(y_test)))

score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
train_score = model.evaluate(np.array(X_train), np.array(y_train))
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], label='Test Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
y_predict = model.predict(np.array(X_test))
"""

#trains using keras and classification
batch_size = 235
num_classes = 4
epochs = 10000

y_train_bin = [dm.calcangle(x[0], x[1]) for x in y_train]
y_test_bin = [dm.calcangle(x[0], x[1]) for x in y_test]

encoder = LabelEncoder()
y_test_enc = encoder.fit_transform(np.array(y_test_bin))
y_train_enc = encoder.transform(np.array(y_train_bin))


model = Sequential()
model.add(Dense(450, activation='relu', input_dim=450))
model.add(Dropout(0.2))
model.add(Dense(450, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(np.array(X_train), y_train_enc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(X_test), y_test_enc))
score = model.evaluate(np.array(X_test), y_test_enc, verbose=0)
y_predict_prob = model.predict(np.array(X_test))
y_predict_enc = [np.argmax(x) for x in y_predict_prob]
y_predict_angle = encoder.inverse_transform(y_predict_enc)
y_predict = [dm.calclabel(x) for x in y_predict_angle]

plt.plot(history.history['loss'], label='Test Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

#calculates the losses from y_predict using the norm of the difference
losses = []
for i in range(len(y_test)):
    loss = dm.quad_single_loss(y_predict[i], y_test[i])
    #print('Loss = ', loss, '\n for file ', file_names[i])
    losses.append(loss)

#calculated the mean and variance of losses and plots them
mean = np.mean(losses)
var = np.var(losses)
print('Mean Loss = ', mean, '   Var Loss', var)
plt.figure()
plt.hist(losses, bins=20)
plt.xlabel('Loss')
plt.ylabel('Number of Samples')
plt.title('Mean: ' + str(mean) + '   Var: ' + str(var))


# scatter plot for predicted points
plt.figure()
plt.scatter([y[0] for y in y_predict], [y[1] for y in y_predict])
plt.axhline(0,color='red') # x = 0
plt.axvline(0,color='red') # y = 0
plt.grid()
plt.show()

