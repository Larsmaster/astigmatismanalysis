import data_methods as dm
import numpy as np
import math as ma
import os as os
import matplotlib.pyplot as plt
import scipy as sp

file_path = 'processed_data'
aim_path = 'analyzed_data'
file_name = ''
file_paths = []

for file in os.listdir(file_path):
    if file.endswith((".csv")) and file.find(file_name) != -1:
        # takes data from specified path with specified name
        file_paths.append(os.path.join(file_path, file))

losses = []
for i,file in enumerate(file_paths):
    print('Processing file ', i, '  of  ' , len(file_paths))
    file_path = file[:-4]
    data = dm.read_file(file)
    values = data[:-3]
    label = data[-3:]
    if label[2] == 0:
        continue
    filename = os.path.basename(file_path)
    #interpolates the values
    x = np.linspace(0,180,len(values))

    def fit_function(x, a, b, c):
        return a * np.cos(2* np.pi/180 * x + c) + b
    try:
        params, params_cov = sp.optimize.curve_fit(fit_function, x, values, p0 = [0.4, 0.75, 0], bounds=([0, -1, -np.inf], [2, 1, np.inf]))
    except RuntimeError:
        print('Did not find optimization')
        continue

    print('Interpolation Parameter: ', params)
    shift = (np.rad2deg(params[2])/2)%180
    print('Shift = ',shift)
    refracerror = 1
    if abs(params[0]) < 0.02:
        refracerror = 0
    plt.title('Calculated:  Phase shift = ' + str(round(shift,4)) + '    Refractive Error (1/0) = ' + str(refracerror) + '\n Ground truth:  Phase shift = '  + str(dm.calcangle(label[0],label[1])) + '    Refractive Error (D)  ' + str(label[2]))
    #plt.title('fitting parameters: \n a = ' +  str(round(params[0],2)) + '  b = ' + str(round(params[1],2)) + '   c = ' + str(round(params[2],2)))

    plt.plot(x, values, label='original data')
    plt.plot(x, fit_function(x, params[0], params[1], params[2]), label='fitted data')
    plt.ylim(0, 1)
    plt.xlabel('angle in degree')
    plt.ylabel('CCA values')
    new_path = os.path.join(aim_path, os.path.basename(file_path))
    plt.savefig(new_path + '.png')
    plt.clf()
    ground_label = (label[0], label[1])
    calc_label = dm.calclabel(shift)
    print(file)
    print('Ground Label = ', ground_label, '   Calc_Label = ', calc_label)
    loss = ma.sqrt(dm.quad_single_loss((label[0], label[1]),(dm.calclabel(shift))))
    print('Loss = ', loss)
    losses.append(loss)


mean = np.mean(losses)
var = np.var(losses)
print('Mean = ', mean, '  Variance = ', var)
plt.hist(losses, bins=20)
plt.xlabel('Loss')
plt.ylabel('Number of Samples')
plt.title('Mean: ' + str(mean) + '   Var: ' + str(var))
plt.show()
