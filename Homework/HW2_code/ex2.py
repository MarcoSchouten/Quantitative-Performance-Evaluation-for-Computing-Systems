import numpy as np
import pandas as pd




def main():
    # read data
    df= pd.read_csv('memory-usage.csv', sep='\t')
    X = df.iloc[:, 0:3].values
    feat_cols = ['size', 'speed', 'inter_arrival']
    df = pd.DataFrame(X,columns=feat_cols)


    # remove outlier
    print('size  before' + str(df.shape))
    idx_to_remove = np.argmin(df.loc[:, 'speed'])   # find min
    df = df.drop([df.index[idx_to_remove]])   # delete the row
    print('size after ' + str(df.shape))



    # get lambda
    arrivals = df.loc[:,'inter_arrival']
    avg_arrival_time = np.average(df.loc[:,'inter_arrival'])
    lambda_ = 1 / avg_arrival_time
    print('lambda: ' +str(lambda_))

    # get mu
    job_size = df.loc[:, 'size']
    speed = df.loc[:, 'speed']
    service_time = np.divide(job_size, speed)
    avg_service_time = np.average(service_time) # avg_service_time= tau
    mu_ = 1 / avg_service_time
    print('mu: ' +str(mu_))

    # compute rho
    rho_ = lambda_ / mu_

    # get Ca
    std_arrivals =  np.std(df.loc[:,'inter_arrival'])
    Ca_ = std_arrivals / avg_arrival_time

    # get Cs
    std_service_time =  np.std(service_time)
    Cs_ =std_service_time / avg_service_time



    #_________________ PART 1 - FORMULAS
    # waiting time for M/M/1
    wait_MM1_formulas = rho_ / (mu_ - lambda_)


    # waiting time for G/G/1
    rho_= lambda_ / mu_
    wait_GG1_formulas = ( rho_ / (1- rho_) )* ((Cs_**2 + Ca_**2)/2) * avg_service_time


    #_________________ PART 2 - SIMULATIONS
    # arrivals
    # service_time

    arrivals = np.array(arrivals)
    service_time = np.array(service_time)


    # init
    n = arrivals.size
    A = np.zeros(n)
    S = np.zeros(n)
    C = np.zeros(n)
    W = np.zeros(n)
    Waitingtime = np.zeros(n)

    # first iteration is hardcoded
    A[0] = arrivals[0]
    S[0] = A[0]
    C[0] = S[0] + service_time[0]
    W[0] = C[0]  -A[0]

    # repeat
    for i in range(1, n):
        A[i] = A[i-1] + arrivals[i]
        S[i] = np.maximum(C[i-1], A[i])
        C[i] = S[i] + service_time[i]
        W[i] = C[i] - A[i]
        Waitingtime[i] = S[i] - A[i]

    wait_GG1_simulation = np.average(Waitingtime)

    wait_MM1_formulas = np.round(wait_MM1_formulas, 3)
    wait_GG1_formulas = np.round(wait_GG1_formulas, 3)
    wait_GG1_simulation = np.round(wait_GG1_simulation, 3)

    print(np.round(wait_GG1_simulation - wait_GG1_formulas, 2) )


    print('wait_MM1_formulas: {} \nwait_GG1_formulas: {} \nwait_GG1_simulation: {} '.format(wait_MM1_formulas,
                                                                                             wait_GG1_formulas, wait_GG1_simulation))

    print('done')





if __name__ == "__main__":
    main()
