import math
import numpy as np
def get_action_space( use_default=True):
    if use_default:
        action_size = 12
        action_space = [np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]),                     # Choice for number of PEs
                        np.array([i + 1 for i in range(action_size)]),                               # Choice for number of buffer unit
                        np.array([1, 2, 3])]                                                         # Choice for dataflows
    else:


        raise NameError("Please define the customized action space.")

    action_bound = [np.max(a) for a in action_space]
    action_bottom = [np.min(a) for a in action_space]
    action_space = [action_space[i] / action_bound[i] for i in range(len(action_space))]
    return action_space, action_bound, action_bottom