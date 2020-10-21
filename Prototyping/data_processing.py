import numpy as np
import math as ma
import os as os
import matplotlib.pyplot as plt
from scipy import signal
import data_methods as dm


#initializes list for files
file_paths = []
file_path = 'data' # the path where the data is stored; for example 'Data'
save_path = 'processed_data'

for file in os.listdir(file_path):
    if file.endswith((".csv")):
        # in case you want to rename something
        #os.rename(os.path.join(file_path, file), os.path.join(file_path,file[:-4]))
        file_paths.append(os.path.join(file_path, file))



while True:
    for (i, file_path) in enumerate(file_paths):
        print(i, '  :  ', file_path)
    i = int(input('Which Dataset do you want to edit? '))
    if (i >= 0 & i < len(file_paths)):
        print('You chose ', file_paths[i])
        values_list = []
        file_path = file_paths[i]
        (values, easy_data) = dm.read_raw_data(file_path[:-4])
        filename = os.path.basename(file_path)[:-4]
        # data is tuple of (filename, values, easy data)
        data = ((filename,values, easy_data))
        # data_norm is tuple of (filename, values)
        data_norm = dm.normalize(data)

        print('Length of data: ', len(data_norm[1]))

        #sc = int(input('Which scaling size? (Preferably 450, 0 means no scaling) '))
        #if(sc):
        data_norm = dm.scale(data_norm,450)
        
        #plt.plot(data_norm[1])
        #plt.title('Mean = ' + str(np.mean(data_norm[1])) + '    Var = ' + str(np.var(data_norm[1])))
        #plt.ylim(0,1)
        #plt.show()
        print(os.path.basename(data_norm[0]))
        referror = float(input('Enter Refractive Power(in diopters): '))
        if referror != 0:
            angle = int(input('Enter Angle(in degree): '))
        else:
            angle = 0
        # adds angle and refractive error to the tuple data_norm
        # data_term is tuple of (filename, values, angle, referror)
        data_term = data_norm + (angle, referror,)
        # data_list is list of all augmented data
        data_list = [data_term]
        #if (int(input('Do you want to split? (1,0) '))):
        #    data1, data2 = dm.split(data_term)
        #    data_list = [data1, data2]

        #shifts the data by input, if 0 -> no shifting
        #sh = int(input('Shift by (degree) ? '))
        #y = 0
        #while y < len(data_list):
        #    data_list[y] = dm.shift(data_list[y], sh)
        #    y = y + 1


        #adds noise to the data
        #if int(input('Add noise? (1/0) ')):
        #    y = 0
        #    while y < len(data_list):
        #        data_list[y] = dm.add_noise(data_list[y], 0.15194, 0.0015788)
        #        y = y + 1


        # saves the data
        for e in data_list:
            # see save function
            # saves data to a figure and csv file
            print('Saving data ', i)
            dm.save_labelled(e, save_path)
        
    else:
        print('File does not exist')







