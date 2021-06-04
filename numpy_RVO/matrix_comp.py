import numpy as np 
from scipy.io import savemat
import os

weight_x = 100
weight_y = 100
# weight_z = 0.0001

paths = os.getcwd()+"/"


def Q_generator(Pddot,nbot):
    x_temp = weight_x*np.dot(Pddot.T,Pddot)
    y_temp = weight_y*np.dot(Pddot.T,Pddot)
    # z_temp = weight_z*np.dot(Pddot.T,Pddot)

    Qx = np.kron(np.eye(nbot,dtype=int),x_temp)
    Qy = np.kron(np.eye(nbot,dtype=int),y_temp)
    # Qz = np.kron(np.eye(nbot,dtype=int),z_temp)

    # return Qx,Qy,Qz
    return Qx,Qy

def Aeq_generator(P,Pdot,Pddot,nbot):
    A = np.vstack((P[0],Pdot[0],Pddot[0],P[-1],Pdot[-1],Pddot[-1]))
    Aeq = np.kron(np.eye(nbot,dtype=int),A)
    return Aeq,Aeq

def Afc_generator(P,nbot,n_samples):
    nbern = 21
    Afc = np.empty((0,nbot*nbern),dtype=float, order='C')
    zero_padding = np.zeros((n_samples,nbern))
    for r in range(1,nbot):
        A_col = np.tile(P,(nbot-r,1))
        A_temp = -np.kron(np.eye(nbot-r,dtype=int),P)
        Ar = np.hstack((np.tile(zero_padding,(nbot-r,r-1)),A_col,A_temp))
        Afc = np.append(Afc,Ar,axis=0)
    
    return Afc

def inv_matrix_generator(nbot,Pddot,Qx,Qy,Aeq_x,Aeq_y,Afc,rho):
    for i in range(len(rho)):
        cost_x = weight_x*np.dot(Pddot.T,Pddot)
        cost_net_x = np.kron(np.eye(nbot,dtype=int),cost_x)
        ele_1 = cost_net_x + rho[i]*np.dot(-Afc.T,-Afc)
        ele_2 = Aeq_x.T 
        ele_3 = Aeq_x
        ele_4 = np.zeros((Aeq_x.shape[0],nbot*6))
        upright_x = np.vstack((np.hstack((ele_1,ele_2)),np.hstack((ele_3,ele_4))))
        inv_x = np.linalg.inv(upright_x)
        savemat(paths+'matrices/cost_mat_inv_x'+str(i)+'.mat', {'inv_x': inv_x})

        cost_y = weight_y*np.dot(Pddot.T,Pddot)
        cost_net_y = np.kron(np.eye(nbot,dtype=int),cost_x)
        ele_1 = cost_net_y + rho[i]*np.dot(-Afc.T,-Afc)
        ele_2 = Aeq_y.T 
        ele_3 = Aeq_y
        ele_4 = np.zeros((Aeq_x.shape[0],nbot*6))
        upright_y = np.vstack((np.hstack((ele_1,ele_2)),np.hstack((ele_3,ele_4))))
        inv_y = np.linalg.inv(upright_y)
        savemat(paths+'matrices/cost_mat_inv_y'+str(i)+'.mat', {'inv_y': inv_y})

        # cost_z = weight_z*np.dot(Pddot.T,Pddot)
        # cost_net_z = np.kron(np.eye(nbot,dtype=int),cost_x)
        # ele_1 = cost_net_z + rho[i]*np.dot(-Afc.T,-Afc)
        # ele_2 = Aeq_z.T 
        # ele_3 = Aeq_z
        # ele_4 = np.zeros((Aeq_x.shape[0],nbot*6))
        # upright_z = np.vstack((np.hstack((ele_1,ele_2)),np.hstack((ele_3,ele_4))))
        # inv_z = np.linalg.inv(upright_z)
        # savemat(paths+'matrices/cost_mat_inv_z'+str(i)+'.mat', {'inv_z': inv_z})

        savemat(paths+'matrices/Aeq_x.mat', {'Aeq_x': Aeq_x})
        savemat(paths+'matrices/Aeq_y.mat', {'Aeq_y': Aeq_y})
        # savemat(paths+'matrices/Aeq_z.mat', {'Aeq_z': Aeq_z})

        savemat(paths+'matrices/Afc.mat', {'Afc': Afc})

    savemat(paths+'matrices/rho.mat', {'rho_init': rho})



        


