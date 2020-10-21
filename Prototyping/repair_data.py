import numpy as np
import os as os

file_paths = []
file_path = 'repair_data' # the path where the data is stored; for example 'Data'

for file in os.listdir(file_path):
    if file.endswith((".easy")):
        # in case you want to rename something
        #os.rename(os.path.join(file_path, file), os.path.join(file_path,file[:-4]))
        file_paths.append(os.path.join(file_path, file))
    if file.endswith((".csv")):
        timestamps = np.genfromtxt(os.path.join(file_path, file), delimiter=",")
        
broken_data = []

for file_path in file_paths:
    broken_data.append((np.genfromtxt(file_path, delimiter='\t'), file_path))

print(np.shape(broken_data[0]))
print(np.shape(timestamps))

for ts in timestamps:
    found_data = False;
    for data in broken_data:
        for da in data[0]:
            if (da[12] - ts[2]) < 2 and (da[12] - ts[2]) >= 0:
                da[11] = ts[1]
                found_data = True;
    print('Found Data: ', found_data, 'for ts ', ts)

for data in broken_data:
    np.savetxt(data[1] + '_repaired', data[0] , delimiter='\t')