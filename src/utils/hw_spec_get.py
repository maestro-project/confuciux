import math
import numpy as np


def get_action_space(is_GEMM=False, action_size=12, is_pe_buf=False):
    if action_size == 12:
        if is_GEMM:
            action_space = [np.array([2 ** (i + 1) for i in range(12)]),
                            np.array([2 ** (i + 1) for i in range(12)]),
                            np.array([1, 2,3])]


        else:
            action_space = [np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
                            np.array([i+1 for i in range(12)]),
                            np.array([1, 2,3])]
            if is_pe_buf:
                action_space = [np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([1, 2, 3])]
    elif action_size ==14:
        if is_GEMM:
            action_space = [np.array([2 ** (i + 1) for i in range(action_size)]),
                            np.array([2 ** (i + 1) for i in range(action_size)]),
                            np.array([1, 2, 3])]
        else:
            action_space = [np.array([1, 2, 4,6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]),
                            np.array([i + 1 for i in range(action_size)]),
                            np.array([1, 2, 3])]
            if is_pe_buf:
                action_space = [np.array([1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([1, 2, 4,6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([1, 2, 3])]
    elif action_size ==10:
        if is_GEMM:
            action_space = [np.array([2 ** (i + 1) for i in range(action_size)]),
                            np.array([2 ** (i + 1) for i in range(action_size)]),
                            np.array([1, 2, 3])]
        else:
            action_space = [np.array([2, 4, 8, 16, 24, 32, 48, 64, 96, 128]),
                            np.array([i + 1 for i in range(action_size)]),
                            np.array([1, 2, 3])]
            if is_pe_buf:
                action_space = [np.array([2, 4, 8, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([2, 4, 8, 16, 24, 32, 48, 64, 96, 128]),
                                np.array([1, 2, 3])]
    action_bound = [np.max(a) for a in action_space]
    action_bottom = [np.min(a) for a in action_space]
    action_space = [action_space[i] / action_bound[i] for i in range(len(action_space))]
    return action_space, action_bound, action_bottom