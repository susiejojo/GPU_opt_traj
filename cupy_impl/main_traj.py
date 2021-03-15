import cupy as cp 
import time,os
from scipy.io import loadmat,savemat
from init_final_pos import init_final_pos
from matrix_comp import Q_generator,Aeq_generator,Afc_generator,inv_matrix_generator
from scipy.special import binom
from bernsteine_20 import bernstein_20_coeffs

# print ("Aeq:",Aeq_x.shape) #96*96
# print ("Afc:",Afc.shape)  #nbot*1000,96

# rho = 7*0.42**2
nbot = 16
a = 0.1

total_time = 10.0
start_time = 0.0
n_samples = 100
time_ints = cp.linspace(start_time,total_time,n_samples)
time_ints = time_ints.reshape(n_samples,1)

P,Pdot,Pddot = bernstein_20_coeffs(time_ints,start_time,total_time)
P = cp.asarray(P)
Pdot = cp.asarray(Pdot)
Pddot = cp.asarray(Pddot)
# Qx,Qy,Qz = Q_generator(Pddot,nbot)
# Aeq_x,Aeq_y,Aeq_z = Aeq_generator(P,Pdot,Pddot,nbot)
# Afc = Afc_generator(P,nbot,n_samples)

# inv_matrix_generator(nbot,Qx,Qy,Qz,Aeq_x,Aeq_y,Aeq_z,Afc,rho)

paths = os.getcwd()+"/drive/MyDrive/GPU_opt_traj/cupy_impl/"
cost_mat_inv_x0 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x0.mat')['inv_x'])
cost_mat_inv_y0 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y0.mat')['inv_y'])
cost_mat_inv_z0 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z0.mat')['inv_z'])

cost_mat_inv_x1 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x1.mat')['inv_x'])
cost_mat_inv_y1 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y1.mat')['inv_y'])
cost_mat_inv_z1 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z1.mat')['inv_z'])

cost_mat_inv_x2 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x2.mat')['inv_x'])
cost_mat_inv_y2 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y2.mat')['inv_y'])
cost_mat_inv_z2 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z2.mat')['inv_z'])

cost_mat_inv_x3 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x3.mat')['inv_x'])
cost_mat_inv_y3 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y3.mat')['inv_y'])
cost_mat_inv_z3 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z3.mat')['inv_z'])

cost_mat_inv_x4 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x4.mat')['inv_x'])
cost_mat_inv_y4 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y4.mat')['inv_y'])
cost_mat_inv_z4 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z4.mat')['inv_z'])

cost_mat_inv_x5 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x5.mat')['inv_x'])
cost_mat_inv_y5 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y5.mat')['inv_y'])
cost_mat_inv_z5 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z5.mat')['inv_z'])

cost_mat_inv_x6 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x6.mat')['inv_x'])
cost_mat_inv_y6 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y6.mat')['inv_y'])
cost_mat_inv_z6 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z6.mat')['inv_z'])

cost_mat_inv_x7 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x7.mat')['inv_x'])
cost_mat_inv_y7 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y7.mat')['inv_y'])
cost_mat_inv_z7 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z7.mat')['inv_z'])

cost_mat_inv_x8 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x8.mat')['inv_x'])
cost_mat_inv_y8 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y8.mat')['inv_y'])
cost_mat_inv_z8 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z8.mat')['inv_z'])

cost_mat_inv_x9 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_x9.mat')['inv_x'])
cost_mat_inv_y9 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_y9.mat')['inv_y'])
cost_mat_inv_z9 = cp.asarray(loadmat(paths+'matrices/cost_mat_inv_z9.mat')['inv_z'])

Aeq_x = cp.asarray(loadmat(paths+'matrices/Aeq_x.mat')['Aeq_x'])
Aeq_y = cp.asarray(loadmat(paths+'matrices/Aeq_y.mat')['Aeq_y'])
Aeq_z = cp.asarray(loadmat(paths+'matrices/Aeq_z.mat')['Aeq_z'])

Afc = cp.asarray(loadmat(paths+'matrices/Afc.mat')['Afc'])

rho = cp.asarray(loadmat(paths+'matrices/rho.mat')['rho_init'][0])
print (rho)

ncomb = int(binom(nbot,2))

x_init, y_init, z_init, x_fin, y_fin, z_fin = init_final_pos()
vx_init = cp.zeros(nbot)
vy_init = cp.zeros(nbot)
vz_init = cp.zeros(nbot)
ax_init = cp.zeros(nbot)
ay_init = cp.zeros(nbot)
az_init = cp.zeros(nbot)

vx_fin = cp.zeros(nbot)
vy_fin = cp.zeros(nbot)
vz_fin = cp.zeros(nbot)
ax_fin = cp.zeros(nbot)
ay_fin = cp.zeros(nbot)
az_fin = cp.zeros(nbot)

beq_x = cp.vstack((x_init,vx_init,ax_init,x_fin,vx_fin,ax_fin)).T #rows = 16, cols = pos,velocity,accln
beq_y = cp.vstack((y_init,vy_init,ay_init,y_fin,vy_fin,ay_fin)).T
beq_z = cp.vstack((z_init,vz_init,az_init,z_fin,vz_fin,az_fin)).T

beq_x = beq_x.reshape(nbot*6) #flatten beqs to (96*1) dimension
beq_y = beq_y.reshape(nbot*6)
beq_z = beq_z.reshape(nbot*6)

alpha_ij = cp.zeros(n_samples*ncomb) #of shape (120*1000,1) as b_fc should be of that shape
beta_ij = cp.zeros(n_samples*ncomb)
d_ij = cp.ones(n_samples*ncomb)
lambda_xij = cp.zeros(n_samples*ncomb)
lambda_yij = cp.zeros(n_samples*ncomb)
lambda_zij = cp.zeros(n_samples*ncomb)
thres = 0.2
b_xfc = cp.zeros(ncomb*n_samples)
b_yfc = b_xfc
b_zfc = b_xfc

def pred_traj(ncomb,b_xfc,b_yfc,b_zfc,rho,a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x,cost_mat_inv_y,cost_mat_inv_z):
    term1 = b_xfc + a*cp.ones(ncomb*n_samples)*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)-lambda_xij/rho
    term2 = b_yfc + a*cp.ones(ncomb*n_samples)*d_ij*cp.sin(alpha_ij)*cp.sin(beta_ij)-lambda_yij/rho
    term3 = b_zfc + a*cp.ones(ncomb*n_samples)*d_ij*cp.cos(beta_ij)-lambda_zij/rho

    aug_term = cp.vstack((term1,term2,term3))
    rhs_top = -rho*cp.dot(Afc.T,aug_term.T).T

    lincost_mat = cp.hstack((-rhs_top, cp.vstack((beq_x,beq_y,beq_z))))

    sol_x = cp.dot(cost_mat_inv_x, lincost_mat[0])
    sol_y = cp.dot(cost_mat_inv_y, lincost_mat[1])
    sol_z = cp.dot(cost_mat_inv_z, lincost_mat[2])
    nvar = 21
    trunc_shape = nbot*nvar

    # sol_x = cp.dot(cost_mat_inv_x, rhs_x)

    # rhs_top = rho*cp.dot(Afc.T,b_yfc)
    # rhs_y = cp.hstack((rhs_top,beq_y))
    # sol_y = cp.dot(cost_mat_inv_y,rhs_y)

    # rhs_top = rho*cp.dot(Afc.T,b_zfc)
    # rhs_z = cp.hstack((rhs_top,beq_z))
    # sol_z = cp.dot(cost_mat_inv_z,rhs_z)
    
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

    x_pred = cp.dot(P,cx.T).T
    y_pred = cp.dot(P,cy.T).T
    z_pred = cp.dot(P,cz.T).T 

    xij = cp.dot(Afc, primal_x)-b_xfc ### xi-xj ### WHYYYYY????????????
    yij = cp.dot(Afc, primal_y)-b_yfc ### yi-yj
    zij = cp.dot(Afc, primal_z)-b_zfc ### zi-zj
 
    alpha_ij = cp.arctan2(yij,xij)
    tij = xij/cp.cos(alpha_ij)
    beta_ij = cp.arctan2(tij,zij)

    c2_d = (lambda_xij*cp.cos(alpha_ij)*cp.sin(beta_ij) + lambda_yij*cp.sin(alpha_ij)*cp.sin(beta_ij) + lambda_zij*cp.cos(beta_ij)+ rho*xij*cp.cos(alpha_ij)*cp.sin(beta_ij) + rho*yij*cp.sin(alpha_ij)*cp.sin(beta_ij) +rho*zij*cp.cos(beta_ij))

    d_temp_1 = c2_d[:ncomb*n_samples]/float(a*rho)
    d_ij = cp.maximum(cp.ones(ncomb*n_samples), d_temp_1)

    res_x = xij-a*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)
    res_y = yij-a*d_ij*cp.sin(alpha_ij)*cp.sin(beta_ij)
    res_z = zij-a*d_ij*cp.cos(beta_ij)

    lambda_xij += rho*res_x
    lambda_yij += rho*res_y
    lambda_zij += rho*res_z

    return x_pred,y_pred,z_pred,res_x,res_y,res_z, lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij

    

n_iters = 500
start = time.time()
for i in range(n_iters):

    print ("Iter #",i)

    if (i<=0.1*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij,xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[0],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x0,cost_mat_inv_y0,cost_mat_inv_z0)

    if (i>0.1*n_iters and i<=0.2*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[1],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x1,cost_mat_inv_y1,cost_mat_inv_z1)

    if (i>0.2*n_iters and i<=0.3*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[2],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x2,cost_mat_inv_y2,cost_mat_inv_z2)

    if (i>0.3*n_iters and i<=0.4*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[3],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x3,cost_mat_inv_y3,cost_mat_inv_z3)

    if (i>0.4*n_iters and i<=0.5*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[4],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x4,cost_mat_inv_y4,cost_mat_inv_z4)

    if (i>0.5*n_iters and i<=0.6*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[5],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x5,cost_mat_inv_y5,cost_mat_inv_z5)

    if (i>0.6*n_iters and i<=0.7*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[6],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x6,cost_mat_inv_y6,cost_mat_inv_z6)

    if (i>0.7*n_iters and i<=0.8*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[7],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x7,cost_mat_inv_y7,cost_mat_inv_z7)

    if (i>0.8*n_iters and i<=0.9*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[8],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x8,cost_mat_inv_y8,cost_mat_inv_z8)

    if (i>0.9*n_iters and i<=n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[9],a,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x9,cost_mat_inv_y9,cost_mat_inv_z9)

    if (cp.max(cp.abs(res_x))<0.02 and cp.max(cp.abs(res_y))<0.02 and cp.max(cp.abs(res_z))<0.02):
        break

    print (cp.max(res_x),cp.max(res_y),cp.max(res_z))
    print (cp.max(res_x)+cp.max(res_y)+cp.max(res_z))

print('comp time for 3 benchmarks =', time.time()-start)
# print (x,y,z)
x = cp.asnumpy(x,stream=None,order='C')
y = cp.asnumpy(y,stream=None,order='C')
z = cp.asnumpy(z,stream=None,order='C')

savemat(paths+'x.mat', {'x': x})
savemat(paths+'y.mat', {'y': y})
savemat(paths+'z.mat', {'z': z})





