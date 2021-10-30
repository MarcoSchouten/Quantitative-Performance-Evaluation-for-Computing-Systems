
import numpy as np
# what I want
# rate E[Nq] / E[N]
# E[Tq]

K = 16

MU = 0.85
RHO = 0.50
LAMBDA = RHO * K * MU




# (0) evaluate Pi_0
sum = 0
for i in range(0, K):
    sum += np.power(K*RHO, i) / np.math.factorial(i) + (np.power(K*RHO, K)) / (np.math.factorial(K) * (1-RHO))
PI_0 = 1 /sum

# (1) evaluate Pq
Pq =  ((np.power(K*RHO, K)) *  PI_0) / (np.math.factorial(K) *(1-RHO))

# (2) evaluate E[Nq] =  Pq * rho / (1-rho)
Nq =   Pq * RHO / (1-RHO)

# (3) evaluate E[Tq] =  E[Nq] / LAMBDA
Tq = Nq /LAMBDA

# (4) evaluate E[T] =  E[Tq] + 1/MU
T = Tq + (1/MU)

# (5) evaluate E[N] =  E[T] * LAMBDA
N = T * LAMBDA

# (6) RATIO = rate E[Nq] / E[N]
RATIO = Nq / N


print('Ratio: {} \n Tq: {}'.format(np.round(RATIO,10), np.round(Tq,10) ))
