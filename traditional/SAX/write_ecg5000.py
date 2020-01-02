from collections import Counter
import numpy as np

import inspect
def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]


data = np.loadtxt('../data/UCR(TRAIN+TEST)/ECG5000.txt', delimiter=',')
all_labels = data[:, 0]
data = data[:, 1:]
rows, cols = data.shape
# print(rows,cols)
#
c = Counter(all_labels)
print(c)
C=list()

# C.append(np.argwhere(all_labels == 1).flatten())
# C.append(np.argwhere(all_labels == 2).flatten())
# C.append(np.argwhere(all_labels == 3).flatten())
# C.append(np.argwhere(all_labels == 4).flatten())
# C.append(np.argwhere(all_labels == 5).flatten())
# print(C[1])

C0 = np.argwhere(all_labels == 1).flatten()
# C1 = np.argwhere(all_labels == 2).flatten()
# C2 = np.argwhere(all_labels == 3).flatten()
# C3 = np.argwhere(all_labels == 4).flatten()
# C4 = np.argwhere(all_labels == 5).flatten()
# print(C1)
for i in range(2,6):
    print(i)
    temp =np.argwhere(all_labels == i).flatten()

    locals()['data' + str(i)] = np.vstack((data[C0, :], data[temp, :]))
    locals()['label' + str(i)] = np.hstack((all_labels[C0], all_labels[temp]))
    np.save('../data/ecg5000/ecg5000_1'+str(i)+'_data', locals()['data' + str(i)])
    np.save('../data/ecg5000/ecg5000_1'+str(i)+'_label', locals()['label' + str(i)])


c = np.load("../data/ecg5000/ecg5000_12_label.npy")
print(c)

