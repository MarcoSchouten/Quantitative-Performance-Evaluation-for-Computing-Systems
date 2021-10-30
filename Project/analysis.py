import pandas as pd
import numpy as np
from datetime import datetime

FMT = '%Y-%m-%dT%H:%M:%SZ'

num_experiments = 32
repetition = 1


def compute_main_effect(y, x, num_experiments):
    return np.dot(x, y) / num_experiments


def sum_square_j(num_experiments, q, repetition):
    return num_experiments * (q ** 2) * repetition


if __name__ == "__main__":
    df = pd.read_csv('log.csv')
    # print(df.tail(5))

    A = np.array(df.loc[:, 'cores'])
    min = A.min()

    for i in range(0, A.size):
        if (A[i] == min):
            A[i] = -1
        else:
            A[i] = 1

    B = np.array(df.loc[:, 'parallelization'])
    min = B.min()
    for i in range(0, B.size):
        if (B[i] == min):
            B[i] = -1
        else:
            B[i] = 1

    C = np.array(df.loc[:, 'batch_size'])
    min = C.min()
    for i in range(0, C.size):
        if (C[i] == min):
            C[i] = -1
        else:
            C[i] = 1

    D = np.array(df.loc[:, 'learning_rate'])
    min = D.min()
    for i in range(0, D.size):
        if (D[i] == min):
            D[i] = -1
        else:
            D[i] = 1
    D = D.astype(int)

    E = np.array(df.loc[:, 'max_epoch'])
    min = E.min()
    for i in range(0, E.size):
        if (E[i] == min):
            E[i] = -1
        else:
            E[i] = 1

    start_time = df.loc[:, 'start_time']
    end_time = df.loc[:, 'end_time']
    y = np.zeros(A.size)
    for i in range(0, start_time.size):
        st_string = start_time.iloc[i]
        st_num = datetime.strptime(st_string, FMT)

        et_string = end_time.iloc[i]
        # print('debug: et_String: {}'.format(et_string))
        et_num = datetime.strptime(et_string, FMT)

        y[i] = (et_num - st_num).seconds

    # built table
    effects_table = pd.DataFrame()
    effects_table['I'] = np.ones(32)
    effects_table['A_cores'] = A
    effects_table['B_parallelization'] = B
    effects_table['C_batch_size'] = C
    effects_table['D_learning_rate'] = D
    effects_table['E_max_epoch'] = E
    effects_table['y_response_time'] = y

    print(effects_table.iloc[:5, :])

    # main effects
    q0 = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'I']),
                             num_experiments)
    qA = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'A_cores']),
                             num_experiments)
    qB = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'B_parallelization']),
                             num_experiments)
    qC = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'C_batch_size']),
                             num_experiments)
    qD = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'D_learning_rate']),
                             num_experiments)
    qE = compute_main_effect(np.array(effects_table.loc[:, 'y_response_time']),
                             np.array(effects_table.loc[:, 'E_max_epoch']),
                             num_experiments)

    # sum of squares
    ss0 = sum_square_j(num_experiments, q0, repetition)
    ssY = np.sum(np.square(np.array(effects_table.loc[:, 'y_response_time'])))
    ssT = ssY - ss0

    ssA = sum_square_j(num_experiments, qA, repetition)
    ssB = sum_square_j(num_experiments, qB, repetition)
    ssC = sum_square_j(num_experiments, qC, repetition)
    ssD = sum_square_j(num_experiments, qD, repetition)
    ssE = sum_square_j(num_experiments, qE, repetition)

    ssError = ssT - num_experiments * repetition * (qA**2 + qB**2 +qC**2 +qD**2 +qE**2)

    # percentage of variation
    variationA = ssA / ssT
    variationB = ssB / ssT
    variationC = ssC / ssT
    variationD = ssD / ssT
    variationE = ssE / ssT
    variationError = ssError / ssT

    print("debug")
