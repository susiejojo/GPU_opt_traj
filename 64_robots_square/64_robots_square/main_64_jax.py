

import jax.numpy as jnp
import numpy as np
import wrapper_64_robot
import optim_jax
from scipy.io import loadmat
import time
from jax import jit
from jax.config import config; config.update("jax_enable_x64", True)
import init_final_pos
import scipy
from scipy.io import loadmat


a_1 = 0.28
b_1 = 0.28
c_1 = 0.28

a_2 = 0.28
b_2 = 0.28
c_2 = 0.28

print('Loading Matrices')

mat_path = "/content/drive/MyDrive/jnp_impl/matrices/"

cost_mat_inv_x_1 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x0.mat')['inv_x'])
cost_mat_inv_x_2 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x1.mat')['inv_x'])
cost_mat_inv_x_3 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x2.mat')['inv_x'])
cost_mat_inv_x_4 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x3.mat')['inv_x'])
cost_mat_inv_x_5 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x4.mat')['inv_x'])
cost_mat_inv_x_6 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x5.mat')['inv_x'])
cost_mat_inv_x_7 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x6.mat')['inv_x'])
cost_mat_inv_x_8 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x7.mat')['inv_x'])
cost_mat_inv_x_9 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x8.mat')['inv_x'])
cost_mat_inv_x_10 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_x9.mat')['inv_x'])

cost_mat_inv_y_1 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y0.mat')['inv_y'])
cost_mat_inv_y_2 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y1.mat')['inv_y'])
cost_mat_inv_y_3 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y2.mat')['inv_y'])
cost_mat_inv_y_4 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y3.mat')['inv_y'])
cost_mat_inv_y_5 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y4.mat')['inv_y'])
cost_mat_inv_y_6 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y5.mat')['inv_y'])
cost_mat_inv_y_7 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y6.mat')['inv_y'])
cost_mat_inv_y_8 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y7.mat')['inv_y'])
cost_mat_inv_y_9 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y8.mat')['inv_y'])
cost_mat_inv_y_10 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_y9.mat')['inv_y'])

cost_mat_inv_z_1 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z0.mat')['inv_z'])
cost_mat_inv_z_2 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z1.mat')['inv_z'])
cost_mat_inv_z_3 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z2.mat')['inv_z'])
cost_mat_inv_z_4 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z3.mat')['inv_z'])
cost_mat_inv_z_5 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z4.mat')['inv_z'])
cost_mat_inv_z_6 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z5.mat')['inv_z'])
cost_mat_inv_z_7 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z6.mat')['inv_z'])
cost_mat_inv_z_8 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z7.mat')['inv_z'])
cost_mat_inv_z_9 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z8.mat')['inv_z'])
cost_mat_inv_z_10 = jnp.asarray(loadmat(mat_path+'cost_mat_inv_z9.mat')['inv_z'])


#
rho_w_alpha_init = jnp.asarray(loadmat(mat_path+'rho.mat')['rho_init'][0])


Ax_eq = jnp.asarray(loadmat(mat_path+'Aeq_x.mat')['Aeq_x'])
# bx_eq = jnp.asarray(loadmat('bx_eq.mat')['bx_eq'])
# by_eq = jnp.asarray(loadmat('by_eq.mat')['by_eq'])
# bz_eq = jnp.asarray(loadmat('bz_eq.mat')['bz_eq'])
Ax_eq_obs = jnp.asarray(loadmat(mat_path+'Afc.mat')['Afc'])

# compute_x_jit = jit(optim_jax.compute_x)
# compute_dobs_jit = jit(optim_jax.compute_dobs)

print('Starting Actual Computation')

x_init, y_init, z_init, x_fin, y_fin, z_fin = init_final_pos.init_final_pos()


start = time.time()
x_1, y_1, z_1 = wrapper_64_robot.main_jax(  x_init, y_init, z_init, x_fin, y_fin, z_fin, a_1, b_1, c_1, cost_mat_inv_x_1, cost_mat_inv_x_2, cost_mat_inv_x_3, cost_mat_inv_x_4, cost_mat_inv_x_5, cost_mat_inv_x_6, cost_mat_inv_x_7, cost_mat_inv_x_8, cost_mat_inv_x_9, cost_mat_inv_x_10, cost_mat_inv_y_1, cost_mat_inv_y_2, cost_mat_inv_y_3, cost_mat_inv_y_4, cost_mat_inv_y_5, cost_mat_inv_y_6, cost_mat_inv_y_7, cost_mat_inv_y_8, cost_mat_inv_y_9, cost_mat_inv_y_10, cost_mat_inv_z_1, cost_mat_inv_z_2, cost_mat_inv_z_3, cost_mat_inv_z_4, cost_mat_inv_z_5, cost_mat_inv_z_6, cost_mat_inv_z_7, cost_mat_inv_z_8, cost_mat_inv_z_9, cost_mat_inv_z_10, Ax_eq_obs, Ax_eq, rho_w_alpha_init     )


# x_2, y_2, z_2 = wrapper_64_robot.main_jax(  x_init, y_init, z_init, x_fin, y_fin, z_fin, a_2, b_2, c_2, cost_mat_inv_x_1, cost_mat_inv_x_2, cost_mat_inv_x_3, cost_mat_inv_x_4, cost_mat_inv_x_5, cost_mat_inv_x_6, cost_mat_inv_x_7, cost_mat_inv_x_8, cost_mat_inv_x_9, cost_mat_inv_x_10, cost_mat_inv_y_1, cost_mat_inv_y_2, cost_mat_inv_y_3, cost_mat_inv_y_4, cost_mat_inv_y_5, cost_mat_inv_y_6, cost_mat_inv_y_7, cost_mat_inv_y_8, cost_mat_inv_y_9, cost_mat_inv_y_10, cost_mat_inv_z_1, cost_mat_inv_z_2, cost_mat_inv_z_3, cost_mat_inv_z_4, cost_mat_inv_z_5, cost_mat_inv_z_6, cost_mat_inv_z_7, cost_mat_inv_z_8, cost_mat_inv_z_9, cost_mat_inv_z_10, Ax_eq_obs, Ax_eq, rho_w_alpha_init     )

print('comp time for 2 benchmarks =', time.time()-start)


scipy.io.savemat('/content/drive/MyDrive/64_robots_square/x_1.mat', {'x_1': x_1})
scipy.io.savemat('/content/drive/MyDrive/64_robots_square/y_1.mat', {'y_1': y_1})
scipy.io.savemat('/content/drive/MyDrive/64_robots_square/z_1.mat', {'z_1': z_1})

# scipy.io.savemat('x_2.mat', {'x_2': x_2})
# scipy.io.savemat('y_2.mat', {'y_2': y_2})
# scipy.io.savemat('z_2.mat', {'z_2': z_2})


print ('done')



