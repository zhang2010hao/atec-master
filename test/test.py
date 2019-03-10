import numpy as np
import torch

lst = np.random.random((2, 3, 2))
print(lst)
print('===============')
indis_1 = [0, 1]
indis_2 = [0, 1]
index_valud = lst[range(2),indis_2, :]
print(index_valud)
print(index_valud.shape)
#
# y = np.random.random((5, 7))
# print(y)
# print(y[2, :])
# o = y[np.array([0, 2, 4]), np.array([0, 1, 2])]
# print(o)
# lst = [[1, 2], [3, 4, 5]]
#
# ten = torch.tensor(lst, dtype=torch.long)
# print(ten)