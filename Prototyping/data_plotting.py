import numpy as np
import os as os
import data_methods as dm


#initializes list for files
file_paths = []
file_path = 'faces_data' # the path where the data is stored; for example 'Data'

for file in os.listdir(file_path):
    if file.endswith((".csv")):
        file_paths.append(os.path.join(file_path, file))


#iterates over all possible files in the folder, so we can process them all in one run if wanted
while True:
    # output to chose a file
    for (i, file_path) in enumerate(file_paths):
        print(i, '  :  ', file_path)
    i = int(input('Which Dataset do you want to edit? '))

    #checks if chosen file exists
    if (i >= 0 & i < len(file_paths)):
        print('You chose ', file_paths[i])
        # reads the cca data from easy and csv file
        file_path = file_paths[i]
        (values, easy_data) = dm.read_raw_data(file_path[:-4])
        #reduces the filepath to filename
        filename = os.path.basename(file_path)[:-4]
        # data is tuple of (filename, values, easy data)
        data = ((filename,values, easy_data))

        # finds position of markers in easy data
        markers_position = np.where(easy_data[:, 11] != 0)[0]

        # manually set for each iteration, markers of faces stimuli
        # with new merge, this can be read from labels
        markers = ['Stranger', 'Pause', 'Stranger', 'Pause', 'Known', 'Pause', 'Known', 'Pause', 'Stranger', 'Pause',
                   'Known', 'Pause', 'Stranger', 'Pause', 'Known']

        # plots and saves the cca values with markers
        dm.plot_raw(data, markers_position=markers_position,markers = markers, save_path='faces_plots')