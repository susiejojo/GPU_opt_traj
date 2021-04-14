from bernsteine_20 import bernstein_20_coeffs
from matrix_comp import Q_generator,Aeq_generator,Afc_generator,inv_matrix_generator
import numpy as np 

total_time = 10.0
start_time = 0.0
n_samples = 100
time_ints = np.linspace(start_time,total_time,n_samples)
time_ints = time_ints.reshape(n_samples,1)
# print (time_ints)

P,Pdot,Pddot = bernstein_20_coeffs(time_ints,start_time,total_time)
print (P.shape,Pdot.shape,Pddot.shape)

n_bot = 16

# rho0 = 7
# rho1 = rho0*4.2
# rho2 = rho0*4.2**2
# rho3 = rho0*4.2**3
# rho4 = rho0*4.2**4
# rho5 = rho0*4.2**5
# rho6 = rho0*6.2**6
# rho7 = rho0*6.2**7
# rho8 = rho0*6.2**8
# rho9 = rho0*6.2**9

rho = [1 for i in range(10)]

# rho = [rho0,rho1,rho2,rho3,rho4,rho5,rho6,rho7,rho8,rho9]

Qx,Qy,Qz = Q_generator(Pddot,n_bot)
Aeq_x,Aeq_y,Aeq_z = Aeq_generator(P,Pdot,Pddot,n_bot)
Afc = Afc_generator(P,n_bot,n_samples)

inv_matrix_generator(n_bot,Pddot,Qx,Qy,Qz,Aeq_x,Aeq_y,Aeq_z,Afc,rho)

