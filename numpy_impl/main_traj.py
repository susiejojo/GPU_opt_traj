import numpy as np 
import time,os
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from init_final_pos import init_final_pos
from matrix_comp import Q_generator,Aeq_generator,Afc_generator,inv_matrix_generator
from scipy.special import binom
from bernsteine_20 import bernstein_20_coeffs
from RVO import avoiding_velocity_sampling,avoiding_velocity_cvxpy

# from jax import jit

total_time = 10.0
start_time = 0.0
n_samples = 100
nvar = 21
time_ints = np.linspace(start_time,total_time,n_samples)
time_ints = time_ints.reshape(n_samples,1)
del_t = time_ints[1] - time_ints[0]
# print (del_t)

P,Pdot,Pddot = bernstein_20_coeffs(time_ints,start_time,total_time)
P = np.asarray(P)
Pdot = np.asarray(Pdot)
Pddot = np.asarray(Pddot)

paths = os.getcwd()+"/"

cost_mat_inv_x0 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x0.mat')['inv_x'])
cost_mat_inv_y0 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y0.mat')['inv_y'])
# cost_mat_inv_z0 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z0.mat')['inv_z'])

cost_mat_inv_x1 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x1.mat')['inv_x'])
cost_mat_inv_y1 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y1.mat')['inv_y'])
# cost_mat_inv_z1 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z1.mat')['inv_z'])

cost_mat_inv_x2 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x2.mat')['inv_x'])
cost_mat_inv_y2 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y2.mat')['inv_y'])
# cost_mat_inv_z2 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z2.mat')['inv_z'])

cost_mat_inv_x3 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x3.mat')['inv_x'])
cost_mat_inv_y3 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y3.mat')['inv_y'])
# cost_mat_inv_z3 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z3.mat')['inv_z'])

cost_mat_inv_x4 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x4.mat')['inv_x'])
cost_mat_inv_y4 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y4.mat')['inv_y'])
# cost_mat_inv_z4 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z4.mat')['inv_z'])

cost_mat_inv_x5 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x5.mat')['inv_x'])
cost_mat_inv_y5 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y5.mat')['inv_y'])
# cost_mat_inv_z5 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z5.mat')['inv_z'])

cost_mat_inv_x6 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x6.mat')['inv_x'])
cost_mat_inv_y6 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y6.mat')['inv_y'])
# cost_mat_inv_z6 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z6.mat')['inv_z'])

cost_mat_inv_x7 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x7.mat')['inv_x'])
cost_mat_inv_y7 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y7.mat')['inv_y'])
# cost_mat_inv_z7 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z7.mat')['inv_z'])

cost_mat_inv_x8 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x8.mat')['inv_x'])
cost_mat_inv_y8 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y8.mat')['inv_y'])
# cost_mat_inv_z8 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z8.mat')['inv_z'])

cost_mat_inv_x9 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_x9.mat')['inv_x'])
cost_mat_inv_y9 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_y9.mat')['inv_y'])
# cost_mat_inv_z9 = np.asarray(loadmat(paths+'matrices/cost_mat_inv_z9.mat')['inv_z'])

Aeq_x = np.asarray(loadmat(paths+'matrices/Aeq_x.mat')['Aeq_x'])
Aeq_y = np.asarray(loadmat(paths+'matrices/Aeq_y.mat')['Aeq_y'])
# Aeq_z = np.asarray(loadmat(paths+'matrices/Aeq_z.mat')['Aeq_z'])

Afc = np.asarray(loadmat(paths+'matrices/Afc.mat')['Afc'])

rho = np.asarray(loadmat(paths+'matrices/rho.mat')['rho_init'][0])
# print (rho)


x_init, y_init, x_fin, y_fin, a, nbot = init_final_pos()
a*=2

ncomb = int(binom(nbot,2))
# print (x_init)
vx_init = np.zeros(nbot)
vy_init = np.zeros(nbot)
# vz_init = np.zeros(nbot)
ax_init = np.zeros(nbot)
ay_init = np.zeros(nbot)
# az_init = np.zeros(nbot)

vx_fin = np.zeros(nbot)
vy_fin = np.zeros(nbot)
# vz_fin = np.zeros(nbot)
ax_fin = np.zeros(nbot)
ay_fin = np.zeros(nbot)
# az_fin = np.zeros(nbot)

beq_x = np.vstack((x_init,vx_init,ax_init,x_fin,vx_fin,ax_fin)).T #rows = 16, cols = pos,velocity,accln
beq_y = np.vstack((y_init,vy_init,ay_init,y_fin,vy_fin,ay_fin)).T
# beq_z = np.vstack((z_init,vz_init,az_init,z_fin,vz_fin,az_fin)).T

beq_x = beq_x.reshape(nbot*6) #flatten beqs to (96*1) dimension
beq_y = beq_y.reshape(nbot*6)
# beq_z = beq_z.reshape(nbot*6)

# print ("beq:",beq_x)

alpha_ij = np.zeros(n_samples*ncomb) #of shape (120*1000,1) as b_fc should be of that shape
# beta_ij = np.zeros(n_samples*ncomb)
d_ij = np.ones(n_samples*ncomb)
lambda_xij = np.zeros(n_samples*ncomb)
lambda_yij = np.zeros(n_samples*ncomb)
# lambda_zij = np.zeros(n_samples*ncomb)
thres = 0.2
b_xfc = np.zeros(ncomb*n_samples)
b_yfc = b_xfc
# b_zfc = b_xfc

def pred_traj(ncomb,nvar,nbot,b_xfc,b_yfc,rho,a,d_ij,alpha_ij,lambda_xij,lambda_yij,Afc,P,beq_x,beq_y,cost_mat_inv_x,cost_mat_inv_y):
    # print (b_xfc.shape)
    # print (lambda_xij.shape)
    term1 = b_xfc + a*np.ones(ncomb*n_samples)*d_ij*np.cos(alpha_ij)-lambda_xij/rho
    term2 = b_yfc + a*np.ones(ncomb*n_samples)*d_ij*np.sin(alpha_ij)-lambda_yij/rho
    # term3 = b_zfc + a*np.ones(ncomb*n_samples)*d_ij*np.cos(beta_ij)-lambda_zij/rho

    aug_term = np.vstack((term1,term2))
    # print (aug_term.shape)
    # print (Afc.shape)
    rhs_top = -rho*np.dot(Afc.T,aug_term.T).T

    lincost_mat = np.hstack((-rhs_top, np.vstack((beq_x,beq_y))))

    sol_x = np.dot(cost_mat_inv_x, lincost_mat[0])
    sol_y = np.dot(cost_mat_inv_y, lincost_mat[1])
    # sol_z = np.dot(cost_mat_inv_z, lincost_mat[2])
    nvar = 21
    trunc_shape = nbot*nvar

    # sol_x = np.dot(cost_mat_inv_x, rhs_x)

    # rhs_top = rho*np.dot(Afc.T,b_yfc)
    # rhs_y = np.hstack((rhs_top,beq_y))
    # sol_y = np.dot(cost_mat_inv_y,rhs_y)

    # rhs_top = rho*np.dot(Afc.T,b_zfc)
    # rhs_z = np.hstack((rhs_top,beq_z))
    # sol_z = np.dot(cost_mat_inv_z,rhs_z)
    
    primal_x = sol_x[:trunc_shape]
    primal_y = sol_y[:trunc_shape]
    # primal_z = sol_z[:trunc_shape]

    # print (primal_y)
    # print (primal_z)

    coeff_x = primal_x[:trunc_shape]
    cx = coeff_x.reshape(nbot,nvar)
    coeff_y = primal_y[:trunc_shape]
    cy = coeff_y.reshape(nbot,nvar)
    # coeff_z = primal_z[:trunc_shape]
    # cz = coeff_z.reshape(nbot,nvar)

    x_pred = np.dot(P,cx.T).T
    y_pred = np.dot(P,cy.T).T
    # z_pred = np.dot(P,cz.T).T 

    vx_pred = np.dot(Pdot,cx.T).T
    vy_pred = np.dot(Pdot,cy.T).T
    # vz_pred = np.dot(Pdot,cz.T).T

    ax_pred = np.dot(Pddot,cx.T).T
    ay_pred = np.dot(Pddot,cy.T).T
    # az_pred = np.dot(Pddot,cz.T).T

    xij = np.dot(Afc, primal_x)-b_xfc ### xi-xj 
    
    yij = np.dot(Afc, primal_y)-b_yfc ### yi-yj
    # zij = np.dot(Afc, primal_z)-b_zfc ### zi-zj
 
    alpha_ij = np.arctan2(yij,xij)
    # tij = xij/np.cos(alpha_ij)
    # beta_ij = np.arctan2(tij,zij)

    c2_d = (lambda_xij*np.cos(alpha_ij) + lambda_yij*np.sin(alpha_ij) + rho*xij*np.cos(alpha_ij)+ rho*yij*np.sin(alpha_ij))

    d_temp_1 = c2_d[:ncomb*n_samples]/(a*rho)
    d_ij = np.maximum(np.ones(ncomb*n_samples), d_temp_1)

    res_x = xij-a*d_ij*np.cos(alpha_ij)
    res_y = yij-a*d_ij*np.sin(alpha_ij)
    # res_z = zij-a*d_ij*np.cos(beta_ij)

    lambda_xij += rho*res_x
    lambda_yij += rho*res_y
    # lambda_zij += rho*res_z

    return x_pred,y_pred, vx_pred, vy_pred,ax_pred, ay_pred, res_x,res_y,lambda_xij,lambda_yij,d_ij,alpha_ij, xij, yij

def recompute_params(nbot,x,y,rho,lambda_xij,lambda_yij,a):
  xij = []
  yij = []
  for i in range(nbot):
    xi = x[i,:]
    yi = y[i,:]
    for j in range(i+1,nbot):
        xj = x[j,:]
        yj = y[j,:]
        xij_pair = xi-xj
        yij_pair = yi-yj
        xij.extend(xij_pair)
        yij.extend(yij_pair)
  xij = np.array(xij)
  # print (xij.shape)
  yij = np.array(yij)
  alpha_ij = np.arctan2(yij,xij)
  c2_d = (lambda_xij*np.cos(alpha_ij) + lambda_yij*np.sin(alpha_ij) + rho*xij*np.cos(alpha_ij)+ rho*yij*np.sin(alpha_ij))
  d_temp_1 = c2_d[:ncomb*n_samples]/(a*rho)
  d_ij = np.maximum(np.ones(ncomb*n_samples), d_temp_1)
  res_x = xij-a*d_ij*np.cos(alpha_ij)
  res_y = yij-a*d_ij*np.sin(alpha_ij)

  lambda_xij += rho*res_x
  lambda_yij += rho*res_y

  return alpha_ij,d_ij,lambda_xij,lambda_yij
        

n_iters = 50
planning_iters = 100
cost_x = []
cost_y = []
# cost_z = []
path_x = []
path_y = []
# path_z = []
comp_time = []

for step in range(planning_iters):
  start = time.time()
  res_x = 0
  res_y = 0
  print ("Running planning step: ",step)

  #planning for 100 steps ahead
  for j in range(n_iters):
    x,y,vx_pred,vy_pred,ax_pred,ay_pred,res_x,res_y,lambda_xij,lambda_yij,d_ij,alpha_ij,xij, yij = pred_traj(ncomb,nvar,nbot,b_xfc,b_yfc,rho[0],a,d_ij,alpha_ij,lambda_xij,lambda_yij,Afc,P,beq_x,beq_y,cost_mat_inv_x0,cost_mat_inv_y0)  
    cost_x.append(np.max(np.abs(res_x)))
    cost_y.append(np.max(np.abs(res_y)))
    
    if (np.max(np.abs(res_x))<0.02 and np.max(np.abs(res_y))<0.02):
      break

  print ("Optimizer converged at ",j,", for planning iter = ",step," with residual: ",np.max(np.abs(res_x)),np.max(np.abs(res_y)))
  # path_x.append(x[:,0])
  # path_y.append(y[:,0])

  V = [[vx_pred[i,0],vy_pred[i,0]] for i in range(nbot)]
  V_des = [[vx_pred[i,1],vy_pred[i,1]] for i in range(nbot)]
  rob_coords = [[x[i,0],y[i,0]] for i in range(nbot)]
  goals = [[x[i,1],y[i,1]] for i in range(nbot)]
  # goals = [[x[i,1],y[i,1]] for i in range(nbot)]
  # print (goals)

  V = np.array(V)
  V_des = np.array(V_des)
  rob_coords = np.array(rob_coords)
  goals = np.array(goals)

  # ##### Sampling ######
  V_change = avoiding_velocity_sampling(rob_coords, goals, V, a/2)
  V = [[(V[i][0]*0.5+V_change[i][0]*0.5),(V[i][1]*0.5+V_change[i][1]*0.5)] for i in range(nbot)]

  # ##### CVXPY ######
  # # vx_new,vy_new = avoiding_velocity_cvxpy(nbot, rob_coords, V, goals, a/2, del_t,V_des)
  # # V = [[vx_new[i],vy_new[i]] for i in range(nbot)]
  V = np.array(V)
  for i in range(nbot):
    vx_pred[i,1] = V[i,0]
    vy_pred[i,1] = V[i,1]
    x[i,1] = x[i,0]+V[i,0]*del_t
    y[i,1] = y[i,0]+V[i,1]*del_t
    ax_pred[i,1] = (vx_pred[i,1]-vx_pred[i,0])/del_t
    ay_pred[i,1] = (vy_pred[i,1]-vy_pred[i,0])/del_t

  alpha_ij,d_ij,lambda_xij,lambda_yij = recompute_params(nbot,x,y,rho[0],lambda_xij,lambda_yij,a)

  beq_x = np.vstack((x[:,1],vx_pred[:,1],ax_pred[:,1],x_fin,vx_fin,ax_fin)).T 
  beq_y = np.vstack((y[:,1],vy_pred[:,1],ay_pred[:,1],y_fin,vy_fin,ay_fin)).T

  beq_x = beq_x.reshape(nbot*6)
  beq_y = beq_y.reshape(nbot*6)
  if (step!=0):
    comp_time.append(time.time()-start)


print ("Mean compute time for each iter:",np.mean(np.array(comp_time)))
fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(111, projection='3d')
plt.plot(cost_x, '-r', linewidth = 3.0)
plt.plot(cost_y, '-b', linewidth = 3.0)
# plt.plot(cost_z, '-g', linewidth = 3.0)
plt.legend(["Res_x","Res_y","Res_z"])
plt.show()


path_x = x
path_y = y

path_x = np.array(path_x)
path_y = np.array(path_y)
print (x.shape)
# path_z = np.array(path_z).T

### Uncomment to find out no of collisions
coll_violate = []
for i in range(nbot):
    x_1 = path_x[i,:]
    y_1 = path_y[i,:]
    # z_1 = path_z[i,:]
    for j in range(nbot):
        if (i!=j):
            x_2 = path_x[j,:]
            y_2 = path_y[j,:]
            dist = np.square((x_2-x_1))+np.square((y_2-y_1))
            coll_violate.append(sum(np.sqrt(dist)<a))
print (sum(coll_violate)//2,"violations out of",ncomb*100)
            
# x = np.asnumpy(x,stream=None,order='C')
# y = np.asnumpy(y,stream=None,order='C')
# z = np.asnumpy(z,stream=None,order='C')

savemat(paths+'x.mat', {'x': path_x})
savemat(paths+'y.mat', {'y': path_y})
# savemat(paths+'z.mat', {'z': path_z})





