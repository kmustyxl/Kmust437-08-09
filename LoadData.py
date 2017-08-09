# -*- coding:utf-8 -*-
import numpy as np
def LoadData(n, m):
    fr = open('test problems\%d_%d dependent.txt'%(n, m))
    data = np.zeros([n, m])
    for line in range(n):
        col = 0
        data_line = fr.readline().split()
        for time in data_line:
            data[line][col] = int(time)
            col += 1
    fr.close()
    return data




