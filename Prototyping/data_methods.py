import numpy as np
import math as ma
import os as os
import matplotlib.pyplot as plt
from scipy import signal


def multilayer_classification(x,y):
    if (x,y) == (0,1):
      return

def quad_single_loss(regr_label, ground_label):
    loss = ma.pow(regr_label[0]-ground_label[0], 2) + ma.pow(regr_label[1] - ground_label[1],2)
    return loss

# shifts the data to the left, can only be applied to scaled data
# data is a tuple of (name, values,angle, refractive error)
def shift(data, angle):
    angle = round(angle)
    if angle != 0:
        print('Shifting data...')
        length = len(data[1])
        left_data = data[1][int(length * angle / 180):]
        right_data = data[1][:int(length * angle / 180)]
        data_value = np.concatenate((left_data,right_data))
        data_name = data[0] + '_shift_' + str(angle)
        data_angle = data[2] + angle
        return (data_name, data_value, data_angle, data[3])
    else:
        return data


# data is a tuple of (name, values, angle, refractive error)
# adds noise to data
def add_noise(data, mean, deviation):
    print('Adding noise...')
    noise = np.random.normal(mean, deviation, len(data[1]))
    data_value = np.add(data[1], noise)
    data_name = data[0] + '_noise'
    return (data_name, data_value, data[2], data[3])


# for 360 degree sweeps the data can be split in two halfs
def split(data):
    print('Splitting data...')
    data1 = (data[0], data[1][:int(len(data[1]) / 2)], data[2], data[3])
    data2 = (data[0] + '_split', data[1][int(len(data[1]) / 2):], data[2], data[3])

    return data1, data2


def save(data, save_path, suffix = ''):

    file_name = os.path.basename(data[0]) + suffix
    file_path = os.path.join(save_path, file_name)
    # saves data into csv file with all (v1, v2, ..., vn, a(angle), b(angle), refractive error)
    try:
        np.savetxt(file_path + '.csv', data[1], delimiter=",")
    except OSError:
        print('Didnt find directory for precessed data. Creating one...')
        os.mkdir(save_path)
        np.savetxt(file_path + '.csv',data[1], delimiter=",")


# saves data to a file and to a plot
def save_labelled(data, save_path, suffix = ''):
    if(data[3] == 0):
        label = (0,0)
    else:
        label = calclabel(data[2])

    # saves the data
    file_name = os.path.basename(data[0]) + suffix
    file_path = os.path.join(save_path, file_name)
    print('Saving data label', label, 'and length', len(data[1]))
    # saves data into csv file with all (v1, v2, ..., vn, a(angle), b(angle), refractive error)
    try:
        np.savetxt(file_path + '.csv', np.concatenate((data[1], list(label), [data[3]])), delimiter=",")
    except OSError:
        print('Didnt find directory for precessed data. Creating one...')
        os.mkdir(save_path)
        np.savetxt(file_path + '.csv', np.concatenate((data[1], list(label), [data[3]])), delimiter=",")

    # saves the plot
    plt.plot(np.linspace(0, 180, len(data[1])), data[1])
    plt.ylim(0, 1)
    plt.title('Lens data: Angle: ' + str(data[2]) + '  Ref Error: ' + str(data[3]) + '\n' + 'Euler Angle: ' + str(
        label[0]) + ' ' + str(label[1]))
    plt.xlabel('Angle in degree')
    plt.ylabel('standartized CCA value')
    plt.savefig(file_path + '.png')
    plt.clf()


# calculates
def calclabel(angle):
    # if refractive error = 0, there is no angle
    # transforms from angle to euler angle
    # rounds to 4 digits
    y = round(np.cos(2 * np.deg2rad(angle)), 4)
    x = round(np.sin(2 * np.deg2rad(angle)), 4)

    return (x,y)

def calcangle(x,y):
    if x == 0:
        if y < 0:
            return 90
        else:
            return 0
    if y == 0:
        if x < 0:
            return 135
        else:
            return 45
    return -np.rad2deg(np.arctan(x/y))*0.5


def scale(data, size):
    print('Scaling data...')
    # using skiyp resampling
    resample_data = data[1]
    if len(data[1]) != size:
        resample_data = signal.resample(data[1], size)
        print('resampled data from length ', len(data[1]), 'to length ', len(resample_data))
        return (data[0] + '_scaled', resample_data)
    return (data[0], resample_data)




# normalizes the data to expectation = 0, variance = 1
# cuts of everything out of start and stop label
def normalize(data):
    print('Normalizing data...')
    # shortens the stimuli by start and stop label
    markers_position = np.where(data[2][:, 11] != 0)[0]
    markers = [int(data[2][m,11]) for m in markers_position]
    for m in markers_position:
        print(int(data[2][m, 11]), ' at ', m)
    if len(markers_position) == 0:
        print('No markers found')

    foundstartstop = False
    while foundstartstop == 0:

        try:
            start = int(np.where(data[2][:,11] == 20000)[0])
            stop = int(np.where(data[2][:,11] == 20180)[0])
        except TypeError:
            start = int(input('StartLabel = '))
            stop = int(input('Stop_Label = '))
        if stop == 0:
            stop = len(data[2])
        plot_raw(data, markers_position=[start, stop], markers=['Start','Stop'])
        foundstartstop = int(input('Passt? (1/0) '))

    # removes the data from values before and after the start and stop label
    value_cut = data[1][round(start / 80): round(stop / 80)]
    # calculates the mean and standart deviation and normalizes the values
    #if (int(input('Normalize Data? (1/0) '))):
    #    value_stdev = np.std(value_cut)
    #    value_mean = np.mean(value_cut)
    #    value_cut = (value_cut - value_mean) / value_stdev
    #    data[0] = data[0] + '_norm'
    return (data[0], value_cut)


# reads the file from data and puts it in an array
def read_raw_data(file_path):
    print('Reading file...')
    # gets all the csv and easy files
    cca_file_path = file_path + '.csv'
    easy_path = file_path + '.easy'
    print('Reading ', easy_path)
    # gets the values
    values = np.genfromtxt(cca_file_path, delimiter=",")

    # get markers
    easy_data = np.genfromtxt(easy_path, delimiter='\t')

    return (values, easy_data)


# plots the data with markers
def plot_raw(data, markers_position=[], markers = [], save_path='', ylabel = 'CCA value', xlabel = 'timesteps'):
    plt.plot(data[1])
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data[0], fontdict=dict(fontsize=8))

    for i, m in enumerate(markers_position):
        plt.axvline(x=m / 80)
        if(len(markers) > i):
            plt.text((m + 250) / 80, 0.9,markers[i], rotation=90,
                     verticalalignment='center')
    if save_path != '':
        file_name = os.path.basename(data[0])
        save_path = os.path.join(save_path, file_name)
        print('Saving raw plot ',file_name)
        plt.savefig(save_path + '_raw.png')
        plt.clf()

    plt.show()




def read_file(file_path):
    print('Reading file...')
    # gets all the csv and easy files
    # gets the values
    values = np.genfromtxt(file_path, delimiter=",")
    return values