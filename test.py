import numpy as np
import torch
import random
ERROR_EPSILON = 0.0001


if __name__ == '__main__':
    memory = []
    for _ in range(32):
        memory.append(torch.tensor([random.uniform(0,1)]))
    print('memory\t', memory)

    sum_abs_error = np.sum(np.absolute(memory))
    sum_abs_error += ERROR_EPSILON * len(memory)
    print('sum_abs_error',sum_abs_error)

    rand_list = np.random.uniform(0, sum_abs_error, 32)
    rand_list = np.sort(rand_list)
    print('rand_list\t', rand_list)

    indexes = []
    idx = 0
    tmp_sum_abs_error = 0

    for rand_num in rand_list:
        print('rand_num\t', rand_num)
        while tmp_sum_abs_error < rand_num:
            tmp_sum_abs_error += (abs(memory[idx])) + ERROR_EPSILON
            print('tmp_sum_abs_error',tmp_sum_abs_error)
            idx += 1

        if idx >= len(memory):
            idx = len(memory) - 1
        indexes.append(idx)

        print('indexes\t', indexes)

    transition = [memory[n] for n in indexes]
    print('transition\t',np.array(transition))
