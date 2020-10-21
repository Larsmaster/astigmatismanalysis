import time
import numpy as np
import os
import glob
import csv
import matplotlib.pyplot as plt

from canonicalca import calc_cca

#pull a file list of all files !!!! you need to edit this for use !!!!
filelist = glob.glob('C:\\local\\Neurophotonics_Hyperopt\\Records\\*\\*.easy')

#defines samples per second
SAMPLES_PER_SECOND = 500

#function for reading the csv file into a numpy array
def read_easy(file):
    with open(file) as csv_file:
        values = np.array(list(csv.reader(csv_file, delimiter='\t')))
        print('Read file and found values with shape:', values.shape)
    return values

#run this for every file in the loop
for file in filelist:
    
    #read data and convert to float values
    data = read_easy(file)
    data = data.astype('float64')

    # find markers and note where they are not zero
    marker_channel = data[:, 11]
    marker = np.where(marker_channel != 0)[0][1:-1]

    # clip eeg channels of used electrodes
    useful_channels = data[:, [0, 2, 3, 4, 5, 6, 7]]

    #do the cca
    xs, result_sum, result_max, score_sum, score_max, sin = calc_cca(useful_channels, 7.5, amount_of_sec=3)


    #plot the wanter results
    plt.plot(xs, result_max)
    plt.ylim(0, 1.0)
    plt.title('Result max')

    #plot all markers
    for m in marker:
        plt.axvline(x=m, color='red')
        
    #save under the same path and name, but as an svg
    pltsave = file[:-5]
    plt.savefig(pltsave + '.svg', format='svg', dpi=1200)
    
    #clear the plot for next iterration
    plt.clf()