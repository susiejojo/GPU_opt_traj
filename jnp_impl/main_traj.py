import jax.numpy as jnp
import time,os
from scipy.io import loadmat,savemat
from init_final_pos import init_final_pos
from matrix_comp import Q_generator,Aeq_generator,Afc_generator,inv_matrix_generator
from scipy.special import binom
from bernsteine_20 import bernstein_20_coeffs
from jax.config import config; config.update("jax_enable_x64", True)

# print ("Aeq:",Aeq_x.shape) #96*96
# print ("Afc:",Afc.shape)  #nbotc2*1000,96

# rho = 7*0.42**2
nbot = 16
a = 0.64
b = 0.64
c = 0.64

total_time = 10.0
start_time = 0.0
n_samples = 100
time_ints = jnp.linspace(start_time,total_time,n_samples)
time_ints = time_ints.reshape(n_samples,1)

P,Pdot,Pddot = bernstein_20_coeffs(time_ints,start_time,total_time)
P = jnp.asarray(P)
Pdot = jnp.asarray(Pdot)
Pddot = jnp.asarray(Pddot)
# Qx,Qy,Qz = Q_generator(Pddot,nbot)
# Aeq_x,Aeq_y,Aeq_z = Aeq_generator(P,Pdot,Pddot,nbot)
# Afc = Afc_generator(P,nbot,n_samples)

# inv_matrix_generator(nbot,Qx,Qy,Qz,Aeq_x,Aeq_y,Aeq_z,Afc,rho)

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
print (rho)

ncomb = int(binom(nbot,2))

x_init, y_init, z_init, x_fin, y_fin, z_fin = init_final_pos()
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

alpha_ij = jnp.zeros(n_samples*ncomb) #of shape (120*1000,1) as b_fc should be of that shape
beta_ij = jnp.zeros(n_samples*ncomb)
d_ij = jnp.ones(n_samples*ncomb)
lambda_xij = jnp.zeros(n_samples*ncomb)
lambda_yij = jnp.zeros(n_samples*ncomb)
lambda_zij = jnp.zeros(n_samples*ncomb)

b_xfc = jnp.zeros(ncomb*n_samples)
b_yfc = b_xfc
b_zfc = b_xfc

def pred_traj(ncomb, b_xfc, b_yfc, b_zfc, rho,a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x,cost_mat_inv_y,cost_mat_inv_z):
    
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

    # sol_x = cp.dot(cost_mat_inv_x, rhs_x)

    # rhs_top = rho*cp.dot(Afc.T,b_yfc)
    # rhs_y = cp.hstack((rhs_top,beq_y))
    # sol_y = cp.dot(cost_mat_inv_y,rhs_y)

    # rhs_top = rho*cp.dot(Afc.T,b_zfc)
    # rhs_z = cp.hstack((rhs_top,beq_z))
    # sol_z = cp.dot(cost_mat_inv_z,rhs_z)

    primal_x = sol_x[:nbot*nvar]
    primal_y = sol_y[:nbot*nvar]
    primal_z = sol_z[:nbot*nvar]

    # print (primal_y)
    # print (primal_z)

    coeff_x = primal_x[:nbot*nvar]
    cx = coeff_x.reshape(nbot,nvar)
    coeff_y = primal_y[:nbot*nvar]
    cy = coeff_y.reshape(nbot,nvar)
    coeff_z = primal_z[:nbot*nvar]
    cz = coeff_z.reshape(nbot,nvar)

    x_pred = jnp.dot(P,cx.T).T
    y_pred = jnp.dot(P,cy.T).T
    z_pred = jnp.dot(P,cz.T).T 

    xij = jnp.dot(Afc, primal_x)-b_xfc ### xi-xj ### WHYYYYY????????????
    yij = jnp.dot(Afc, primal_y)-b_yfc ### yi-yj
    zij = jnp.dot(Afc, primal_z)-b_zfc ### zi-zj
 
    alpha_ij = jnp.arctan2(yij,xij)
    tij = xij/jnp.cos(alpha_ij)
    beta_ij = jnp.arctan2(tij,zij)

    c1_d = 1.0*a**2*rho
    c2_d = 1.0*a*(lambda_xij*jnp.cos(alpha_ij)*jnp.sin(beta_ij) + lambda_yij*jnp.sin(alpha_ij)*jnp.sin(beta_ij) + lambda_zij*jnp.cos(beta_ij)+ rho*xij*jnp.cos(alpha_ij)*jnp.sin(beta_ij) + rho*yij*jnp.sin(alpha_ij)*jnp.sin(beta_ij) +rho*zij*jnp.cos(beta_ij))


    d_temp_1 = c2_d[:ncomb*n_samples]/c1_d
    d_temp = d_temp_1
    d_ij = jnp.maximum(jnp.ones(ncomb*n_samples), d_temp)

    res_x = xij-a*d_ij*jnp.cos(alpha_ij)*jnp.sin(beta_ij)
    res_y = yij-a*d_ij*jnp.sin(alpha_ij)*jnp.sin(beta_ij)
    res_z = zij-a*d_ij*jnp.cos(beta_ij)

    lambda_xij = lambda_xij +res_x*rho
    lambda_yij = lambda_yij +res_y*rho
    lambda_zij = lambda_zij +res_z*rho

    return x_pred,y_pred,z_pred,res_x,res_y,res_z, lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij

    

n_iters = 500
start = time.time()
for i in range(n_iters):
    # b_xfc = lxy*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)-lambda_xij/rho
    # b_yfc = lxy*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)-lambda_yij/rho
    # b_zfc = lz*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)-lambda_zij/rho
    # rhs_top = rho*cp.dot(Afc.T,b_xfc)
    # rhs_x = cp.hstack((rhs_top,beq_x))

    # # print ("LHS:",cost_mat_inv_x.shape)
    # # print ("RHS:",rhs_x.shape)
    # print ("Iter #",i)
    # sol_x = cp.dot(cost_mat_inv_x, rhs_x)

    # rhs_top = rho*cp.dot(Afc.T,b_yfc)
    # rhs_y = cp.hstack((rhs_top,beq_y))
    # sol_y = cp.dot(cost_mat_inv_y,rhs_y)

    # rhs_top = rho*cp.dot(Afc.T,b_zfc)
    # rhs_z = cp.hstack((rhs_top,beq_z))
    # sol_z = cp.dot(cost_mat_inv_z,rhs_z)

    # coeff_x = sol_x[:nbot*6]
    # cx = coeff_x.reshape(nbot,6)
    # coeff_y = sol_y[:nbot*6]
    # cy = coeff_y.reshape(nbot,6)
    # coeff_z = sol_z[:nbot*6]
    # cz = coeff_z.reshape(nbot,6)

    # x_pred = cp.dot(P,cx.T).T
    # y_pred = cp.dot(P,cy.T).T
    # z_pred = cp.dot(P,cz.T).T 

    # xij = cp.dot(Afc, coeff_x)-b_xfc ### xi-xj ### WHYYYYY????????????
    # yij = cp.dot(Afc, coeff_y)-b_yfc ### yi-yj
    # zij = cp.dot(Afc, coeff_z)-b_zfc ### zi-zj
 
    # alpha_ij = cp.arctan2(yij,xij)
    # beta_ij = cp.arctan2(xij/(lxy*cp.cos(alpha_ij)),zij/lz)

    # c1_d = 1.0*lxy**2*rho
    # c2_d = 1.0*lxy*(lambda_xij*cp.cos(alpha_ij)*cp.sin(beta_ij) + lambda_yij*cp.sin(alpha_ij)*cp.sin(beta_ij) + lambda_zij*cp.cos(beta_ij)+ rho*xij*cp.cos(alpha_ij)*cp.sin(beta_ij) + rho*yij*cp.sin(alpha_ij)*cp.sin(beta_ij) +rho*zij*cp.cos(beta_ij))


    # d_temp_1 = c2_d[:ncomb*n_samples]/c1_d
    # d_temp = d_temp_1
    # d_ij = cp.maximum(cp.ones((ncomb)*n_samples), d_temp)

    # res_x = xij-lxy*d_ij*cp.cos(alpha_ij)*cp.sin(beta_ij)
    # res_y = yij-lxy*d_ij*cp.sin(alpha_ij)*cp.sin(beta_ij)
    # res_z = zij-lz*d_ij*cp.cos(beta_ij)

    # lambda_xij = lambda_xij +res_x*rho
    # lambda_yij = lambda_yij +res_y*rho
    # lambda_zij = lambda_zij +res_z*rho
    print ("Iter #",i)

    if (i<=0.1*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij,xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[0],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x0,cost_mat_inv_y0,cost_mat_inv_z0)

    if (i>0.1*n_iters and i<=0.2*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[1],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x1,cost_mat_inv_y1,cost_mat_inv_z1)

    if (i>0.2*n_iters and i<=0.3*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij,alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[2],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x2,cost_mat_inv_y2,cost_mat_inv_z2)

    if (i>0.3*n_iters and i<=0.4*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij,beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[3],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x3,cost_mat_inv_y3,cost_mat_inv_z3)

    if (i>0.4*n_iters and i<=0.5*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[4],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x4,cost_mat_inv_y4,cost_mat_inv_z4)

    if (i>0.5*n_iters and i<=0.6*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[5],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x5,cost_mat_inv_y5,cost_mat_inv_z5)

    if (i>0.6*n_iters and i<=0.7*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[6],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x6,cost_mat_inv_y6,cost_mat_inv_z6)

    if (i>0.7*n_iters and i<=0.8*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[7],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x7,cost_mat_inv_y7,cost_mat_inv_z7)

    if (i>0.8*n_iters and i<=0.9*n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[8],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x8,cost_mat_inv_y8,cost_mat_inv_z8)

    if (i>0.9*n_iters and i<=n_iters):
        x,y,z,res_x,res_y,res_z,lambda_xij,lambda_yij,lambda_zij,d_ij, alpha_ij, beta_ij, xij, yij, zij, tij = pred_traj(ncomb, b_xfc,b_yfc,b_zfc, rho[9],a,b,c,d_ij,alpha_ij,beta_ij,lambda_xij,lambda_yij,lambda_zij,Afc,P,beq_x,beq_y,beq_z,cost_mat_inv_x9,cost_mat_inv_y9,cost_mat_inv_z9)

    if (jnp.max(jnp.abs(res_x))<0.02 and jnp.max(jnp.abs(res_y))<0.02 and jnp.max(jnp.abs(res_z))<0.02):
        break

    print (jnp.max(res_x),jnp.max(res_y),jnp.max(res_z))

print('comp time for 3 benchmarks =', time.time()-start)
# print (x,y,z)
# x = jnp.asnumpy(x,stream=None,order='C')
# y = jnp.asnumpy(y,stream=None,order='C')
# z = jnp.asnumpy(z,stream=None,order='C')

savemat(paths+'x.mat', {'x': x})
savemat(paths+'y.mat', {'y': y})
savemat(paths+'z.mat', {'z': z})





