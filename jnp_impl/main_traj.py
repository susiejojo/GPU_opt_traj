import jax.numpy as jnp 
import time,os
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from init_final_pos import init_final_pos
from matrix_comp import Q_generator,Aeq_generator,Afc_generator,inv_matrix_generator
from scipy.special import binom
from bernsteine_20 import bernstein_20_coeffs
from jax.config import config
config.update("jax_enable_x64", True)

from jax import jit

total_time = 10.0
start_time = 0.0
n_samples = 100
nvar = 21
time_ints = jnp.linspace(start_time,total_time,n_samples)
time_ints = time_ints.reshape(n_samples,1)

P,Pdot,Pddot = bernstein_20_coeffs(time_ints,start_time,total_time)
P = jnp.asarray(P)
Pdot = jnp.asarray(Pdot)
Pddot = jnp.asarray(Pddot)

paths = os.getcwd()+"/drive/MyDrive/GPU_opt_traj/jnp_impl/"
cost_mat_inv_x0 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x0.mat')['inv_x'])
cost_mat_inv_y0 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y0.mat')['inv_y'])
cost_mat_inv_z0 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z0.mat')['inv_z'])

cost_mat_inv_x1 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x1.mat')['inv_x'])
cost_mat_inv_y1 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y1.mat')['inv_y'])
cost_mat_inv_z1 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z1.mat')['inv_z'])

cost_mat_inv_x2 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x2.mat')['inv_x'])
cost_mat_inv_y2 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y2.mat')['inv_y'])
cost_mat_inv_z2 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z2.mat')['inv_z'])

cost_mat_inv_x3 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x3.mat')['inv_x'])
cost_mat_inv_y3 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y3.mat')['inv_y'])
cost_mat_inv_z3 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z3.mat')['inv_z'])

cost_mat_inv_x4 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x4.mat')['inv_x'])
cost_mat_inv_y4 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y4.mat')['inv_y'])
cost_mat_inv_z4 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z4.mat')['inv_z'])

cost_mat_inv_x5 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x5.mat')['inv_x'])
cost_mat_inv_y5 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y5.mat')['inv_y'])
cost_mat_inv_z5 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z5.mat')['inv_z'])

cost_mat_inv_x6 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x6.mat')['inv_x'])
cost_mat_inv_y6 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y6.mat')['inv_y'])
cost_mat_inv_z6 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z6.mat')['inv_z'])

cost_mat_inv_x7 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x7.mat')['inv_x'])
cost_mat_inv_y7 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y7.mat')['inv_y'])
cost_mat_inv_z7 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z7.mat')['inv_z'])

cost_mat_inv_x8 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x8.mat')['inv_x'])
cost_mat_inv_y8 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y8.mat')['inv_y'])
cost_mat_inv_z8 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z8.mat')['inv_z'])

cost_mat_inv_x9 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_x9.mat')['inv_x'])
cost_mat_inv_y9 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_y9.mat')['inv_y'])
cost_mat_inv_z9 = jnp.asarray(loadmat(paths+'matrices/cost_mat_inv_z9.mat')['inv_z'])

Aeq_x = jnp.asarray(loadmat(paths+'matrices/Aeq_x.mat')['Aeq_x'])
Aeq_y = jnp.asarray(loadmat(paths+'matrices/Aeq_y.mat')['Aeq_y'])
Aeq_z = jnp.asarray(loadmat(paths+'matrices/Aeq_z.mat')['Aeq_z'])

Afc = jnp.asarray(loadmat(paths+'matrices/Afc.mat')['Afc'])

rho = jnp.asarray(loadmat(paths+'matrices/rho.mat')['rho_init'][0])
# print (rho)


x_init, y_init, z_init, x_fin, y_fin, z_fin, a, nbot = init_final_pos()
a*=2

ncomb = int(binom(nbot,2))
# print (x_init)
vx_init = jnp.zeros(nbot)
vy_init = jnp.zeros(nbot)
vz_init = jnp.zeros(nbot)
ax_init = jnp.zeros(nbot)
ay_init = jnp.zeros(nbot)
az_init = jnp.zeros(nbot)

vx_fin = jnp.zeros(nbot)
vy_fin = jnp.zeros(nbot)
vz_fin = jnp.zeros(nbot)
ax_fin = jnp.zeros(nbot)
ay_fin = jnp.zeros(nbot)
az_fin = jnp.zeros(nbot)

beq_x = jnp.vstack((x_init,vx_init,ax_init,x_fin,vx_fin,ax_fin)).T #rows = 16, cols = pos,velocity,accln
beq_y = jnp.vstack((y_init,vy_init,ay_init,y_fin,vy_fin,ay_fin)).T
beq_z = jnp.vstack((z_init,vz_init,az_init,z_fin,vz_fin,az_fin)).T

beq_x = beq_x.reshape(nbot*6) #flatten beqs to (96*1) dimension
beq_y = beq_y.reshape(nbot*6)
beq_z = beq_z.reshape(nbot*6)

# print ("beq:",beq_x)

alpha_ij = jnp.zeros(n_samples*ncomb) #of shape (120*1000,1) as b_fc should be of that shape
beta_ij = jnp.zeros(n_samples*ncomb)
d_ij = jnp.ones(n_samples*ncomb)
lambda_xij = jnp.zeros(n_samples*ncomb)
lambda_yij = jnp.zeros(n_samples*ncomb)
lambda_zij = jnp.zeros(n_samples*ncomb)
thres = 0.2
b_xfc = jnp.zeros(ncomb*n_samples)
b_yfc = b_xfc
b_zfc = b_xfc

def pred_traj(ncomb,nvar,nbot,b_xfc,b_yfc,b_zfc,rho,a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x,cost_mat_inv_y,cost_mat_inv_z):
    term1 = b_xfc + a*jnp.ones(ncomb*n_samples)*d_ij*jnp.cos(alpha_ij)*jnp.sin(beta_ij)-lambda_xij/rho
    term2 = b_yfc + a*jnp.ones(ncomb*n_samples)*d_ij*jnp.sin(alpha_ij)*jnp.sin(beta_ij)-lambda_yij/rho
    term3 = b_zfc + a*jnp.ones(ncomb*n_samples)*d_ij*jnp.cos(beta_ij)-lambda_zij/rho

    aug_term = jnp.vstack((term1,term2,term3))
    rhs_top = -rho*jnp.dot(Afc.T,aug_term.T).T

    lincost_mat = jnp.hstack((-rhs_top, jnp.vstack((beq_x,beq_y,beq_z))))

    sol_x = jnp.dot(cost_mat_inv_x, lincost_mat[0])
    sol_y = jnp.dot(cost_mat_inv_y, lincost_mat[1])
    sol_z = jnp.dot(cost_mat_inv_z, lincost_mat[2])
    nvar = 21
    trunc_shape = nbot*nvar

    # sol_x = jnp.dot(cost_mat_inv_x, rhs_x)

    # rhs_top = rho*jnp.dot(Afc.T,b_yfc)
    # rhs_y = jnp.hstack((rhs_top,beq_y))
    # sol_y = jnp.dot(cost_mat_inv_y,rhs_y)

    # rhs_top = rho*jnp.dot(Afc.T,b_zfc)
    # rhs_z = jnp.hstack((rhs_top,beq_z))
    # sol_z = jnp.dot(cost_mat_inv_z,rhs_z)
    
    primal_x = sol_x[:trunc_shape]
    primal_y = sol_y[:trunc_shape]
    primal_z = sol_z[:trunc_shape]

    # print (primal_y)
    # print (primal_z)

    coeff_x = primal_x[:trunc_shape]
    cx = coeff_x.reshape(nbot,nvar)
    coeff_y = primal_y[:trunc_shape]
    cy = coeff_y.reshape(nbot,nvar)
    coeff_z = primal_z[:trunc_shape]
    cz = coeff_z.reshape(nbot,nvar)

    x_pred = jnp.dot(P,cx.T).T
    y_pred = jnp.dot(P,cy.T).T
    z_pred = jnp.dot(P,cz.T).T 

    vx_pred = jnp.dot(Pdot,cx.T).T
    vy_pred = jnp.dot(Pdot,cy.T).T
    vz_pred = jnp.dot(Pdot,cz.T).T

    ax_pred = jnp.dot(Pddot,cx.T).T
    ay_pred = jnp.dot(Pddot,cy.T).T
    az_pred = jnp.dot(Pddot,cz.T).T

    xij = jnp.dot(Afc, primal_x)-b_xfc ### xi-xj ### WHYYYYY????????????
    yij = jnp.dot(Afc, primal_y)-b_yfc ### yi-yj
    zij = jnp.dot(Afc, primal_z)-b_zfc ### zi-zj
 
    alpha_ij = jnp.arctan2(yij,xij)
    tij = xij/jnp.cos(alpha_ij)
    beta_ij = jnp.arctan2(tij,zij)

    c2_d = (lambda_xij*jnp.cos(alpha_ij)*jnp.sin(beta_ij) + lambda_yij*jnp.sin(alpha_ij)*jnp.sin(beta_ij) + lambda_zij*jnp.cos(beta_ij)+ rho*xij*jnp.cos(alpha_ij)*jnp.sin(beta_ij) + rho*yij*jnp.sin(alpha_ij)*jnp.sin(beta_ij) +rho*zij*jnp.cos(beta_ij))

    d_temp_1 = c2_d[:ncomb*n_samples]/(a*rho)
    d_ij = jnp.maximum(jnp.ones(ncomb*n_samples), d_temp_1)

    res_x = xij-a*d_ij*jnp.cos(alpha_ij)*jnp.sin(beta_ij)
    res_y = yij-a*d_ij*jnp.sin(alpha_ij)*jnp.sin(beta_ij)
    res_z = zij-a*d_ij*jnp.cos(beta_ij)

    lambda_xij += rho*res_x
    lambda_yij += rho*res_y
    lambda_zij += rho*res_z

    return x_pred,y_pred,z_pred, vx_pred, vy_pred, vz_pred, ax_pred, ay_pred, az_pred, res_x,res_y,res_z, lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij

    

n_iters = 1
planning_iters = 1000
cost_x = []
cost_y = []
cost_z = []
path_x = []
path_y = []
path_z = []
comp_time = []

for step in range(planning_iters):
  start = time.time()

    #planning for 100 steps ahead
  for i in range(n_iters):
      pred_traj_jit = jit(pred_traj,static_argnums=(0,1,2,))
      x,y,z,vx_pred,vy_pred,vz_pred,ax_pred,ay_pred,az_pred,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij,xij, yij, zij, tij = pred_traj_jit(ncomb,nvar,nbot,b_xfc,b_yfc,b_zfc, rho[0],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x0,cost_mat_inv_y0,cost_mat_inv_z0)
          
      cost_x.append(jnp.max(jnp.abs(res_x)))
      cost_y.append(jnp.max(jnp.abs(res_y)))
      cost_z.append(jnp.max(jnp.abs(res_z)))

      # if (jnp.max(jnp.abs(res_x))<0.02 and jnp.max(jnp.abs(res_y))<0.02 and jnp.max(jnp.abs(res_z))<0.02):
      #     break
  if (step!=0):
    comp_time.append(time.time()-start)
  # print('comp time for 1 iter =', time.time()-start)
  path_x.append(x[:,0])
  # print (x.shape)
  # print (x[:,0])
  path_y.append(y[:,0])
  path_z.append(z[:,0])
  beq_x = jnp.vstack((x[:,1],vx_pred[:,1],ax_pred[:,1],x_fin,vx_fin,ax_fin)).T 
  beq_y = jnp.vstack((y[:,1],vy_pred[:,1],ay_pred[:,1],y_fin,vy_fin,ay_fin)).T
  beq_z = jnp.vstack((z[:,1],vz_pred[:,1],az_pred[:,1],z_fin,vz_fin,az_fin)).T

  beq_x = beq_x.reshape(nbot*6)
  beq_y = beq_y.reshape(nbot*6)
  beq_z = beq_z.reshape(nbot*6)
  # print (beq_x)

  # print (jnp.max(jnp.abs(res_x)),jnp.max(jnp.abs(res_y)),jnp.max(jnp.abs(res_z)))
  # print (jnp.max(res_x)+jnp.max(res_y)+jnp.max(res_z))
print ("Mean compute time for each iter:",jnp.mean(jnp.array(comp_time)))
fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(111, projection='3d')
plt.plot(cost_x, '-r', linewidth = 3.0)
plt.plot(cost_y, '-b', linewidth = 3.0)
plt.plot(cost_z, '-g', linewidth = 3.0)
plt.legend(["Res_x","Res_y","Res_z"])
plt.show()

# print (path_x)
path_x.append(x_fin)
path_y.append(y_fin)
path_z.append(z_fin)
# print (x,y,z)
path_x = jnp.array(path_x).T
path_y = jnp.array(path_y).T
path_z = jnp.array(path_z).T

coll_violate = []
for i in range(nbot):
    x_1 = path_x[i,:]
    y_1 = path_y[i,:]
    z_1 = path_z[i,:]
    for j in range(nbot):
        if (i!=j):
            x_2 = path_x[j,:]
            y_2 = path_y[j,:]
            z_2 = path_z[j,:]
            dist = jnp.square((x_2-x_1))+jnp.square((y_2-y_1))+jnp.square((z_2-z_1))
            coll_violate.append(sum(jnp.sqrt(dist)<a))
            # print ("Distance between agents at times: ",i," and ",j," = ",min(jnp.sqrt(dist)))
print (sum(coll_violate)//2,"violations out of",ncomb*100)
            
# x = jnp.asnumpy(x,stream=None,order='C')
# y = jnp.asnumpy(y,stream=None,order='C')
# z = jnp.asnumpy(z,stream=None,order='C')

savemat(paths+'x.mat', {'x': path_x})
savemat(paths+'y.mat', {'y': path_y})
savemat(paths+'z.mat', {'z': path_z})





