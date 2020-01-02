# -*- coding: utf-8 -*-
'''

'''

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

data = np.loadtxt('../data/UCR(TRAIN+TEST)/ECG5000.txt', delimiter=',')
labels = data[:, 0]
data = data[:, 1:]
rows, cols = data.shape
print(rows, cols)
print(Counter(labels))

# data = np.loadtxt('../data/UCR(TRAIN+TEST)/ECG200.txt', delimiter=',')
# labels = data[:, 0]
# print(Counter(labels))
#
# data = np.loadtxt('../data/UCR(TRAIN+TEST)/ECGFiveDays.txt', delimiter=',')
# labels = data[:, 0]
# print(Counter(labels))


c = Counter(labels)
print(c)
C1 = np.argwhere(labels == 1).flatten()
C2 = np.argwhere(labels == 2).flatten()
C3 = np.argwhere(labels == 3).flatten()
C4 = np.argwhere(labels == 4).flatten()
C5 = np.argwhere(labels == 5).flatten()
# print(C1)
# print(C2)
# print(C3)
# print(C4)
# print(C5)
x = np.arange(0, 140)
plt.figure()  # Second, PAA
plt.plot(x, data[C1, :][0], "b-", alpha=0.4,label='N')

plt.plot(x, data[C4, :][0], "r-", alpha=0.4,label='V')

plt.title("PAA")
plt.legend(loc='upper right')
plt.tight_layout()
# plt.savefig('compareECG.png')
plt.show()
