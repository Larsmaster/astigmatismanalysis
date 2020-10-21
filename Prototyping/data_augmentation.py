import numpy as np
import os as os
import matplotlib.pyplot as plt
import data_methods as dm
import math as ma

#initializes list for files
file_paths = []
file_path = 'processed_data' # the path where the data is stored; for example 'Data'
save_path = 'augmented_data'

for file in os.listdir(file_path):
    if file.endswith((".csv")):
        # in case you want to rename something
        #os.rename(os.path.join(file_path, file), os.path.join(file_path,file[:-4]))
        file_paths.append(os.path.join(file_path, file))
total_data = []
for i,file in enumerate(file_paths):
    print('Augmenting file ', i, ' of ', len(file_paths) , ' files' )
    raw_data = dm.read_file(file)
    values = raw_data[:-3]
    label = raw_data[-3:]
    data = (file[:-4], values, dm.calcangle(label[0], label[1]), label[2])
    augmented_data = []
    #shifting data
    shifts = [0, 45, 90, 135]
    for sh in shifts:
        augmented_data.append(dm.shift(data, sh))
    # adds noise
    data_noise = []
    for d in augmented_data:
        data_noise.append(dm.add_noise(d,0,ma.sqrt(0.0015788)))
    augmented_data = augmented_data + data_noise
    total_data = total_data + augmented_data

for data in total_data:
    dm.save(data, 'augmented_data')

