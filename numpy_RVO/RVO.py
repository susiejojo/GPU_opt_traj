import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cvxpy as cp 
import matplotlib
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib.cm as cmx
import matplotlib.colors as colors


def avoiding_velocity_cvxpy(nbot, cur_pos,cur_velo,robot_goals,robot_rad,del_t,V_des):
    agent_velo_x = []
    agent_velo_y = []
    start = time.time()
    for i in range(nbot):
        vA = cp.Variable((2,1))
        v_des = V_des[i].reshape((2,1))
        vA0 = cur_velo[i].reshape((2,1))
        pA = cur_pos[i].reshape((2,1))
        for j in range(nbot):
            if (i!=j):
                vB = cur_velo[j].reshape((2,1))
                pB = cur_pos[j].reshape((2,1))
                vAB = (vA-vB)
                pAB = (pA-pB).reshape((2,1))
                dotp = cp.multiply(vAB.T,pAB)**2
                # print (dotp.shape)
                R = (2*robot_rad)
                # print (R)
                norm_pAB = np.linalg.norm(pAB)**2
                norm_vAB = cp.norm(vAB)**2
                term1 = ((vA0-vB)**2).T@((pAB+vA0*del_t)**2)+((vA0-vB)**2).T@((R**2-(pAB+vA0*del_t)**2))
                # print (term1[0][0])
                term2 = 2*R**2*(vA0-vB)
                # print (term2[0][0])
                constr = [ vA>=0 , (term1+(vA-vA0).T@term2)<=0]
                # print (-cp.norm(pAB)**2)
                # constr += [-np.linalg.norm(pAB)**2 + np.dot(pAB,vAB)**2*np.linalg.norm(vAB)**2 <= -(2*robot_rad)**2]
                # goal_obj = holo_kinematics(pA, vA, robot_goals[i], del_t)
                goal = robot_goals[i].reshape((2,1))
                goal_obj = cp.Minimize((cp.norm(vA - v_des))**2)
                prob = cp.Problem(goal_obj,constr)
                v_reqd = prob.solve()
                # print (vA[0].value,vA[1].value,vA[2].value)
                agent_velo_x.append(vA[0].value)
                agent_velo_y.append(vA[1].value)
    print ("Time:",time.time()-start)
    return agent_velo_x,agent_velo_y

def distance(pose1, pose2):
    return math.sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)

def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= math.pi:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left <0) and (theta_right >0):
            theta_left += 2*math.pi
            if theta_dif < 0:
                theta_dif += 2*math.pi
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left >0) and (theta_right <0):
            theta_right += 2*math.pi
            if theta_dif < 0:
                theta_dif += 2*math.pi
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False

def compute_V_des(X, goal, V_max):
    V_des = []
    nbot = len(X)
    # print (nbot)
    for i in range(len(X)):
        dif_x = [goal[i][k]-X[i][k] for k in range(2)]
        norm = distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.1):
            V_des[i][0] = 0
            V_des[i][1] = 0
    return V_des
            
def reach(p1, p2, bound=0.5):
    if distance(p1,p2)< bound:
        return True
    else:
        return False

def velo_checker(pA, vA, RVO_BA_all):
    norm_v = distance(vA, [0, 0])
    if (norm_v<0.01):
        return vA,0
    suitable_V = []
    ang_v = []
    unsuitable_V = []
    rad_s = 0
    for theta in np.arange(0, 2*math.pi, 0.1):
        for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
            new_v = [rad*math.cos(theta), rad*math.sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
                theta_dif = math.atan2(dif[1], dif[0])
                theta_right = math.atan2(right[1], right[0])
                theta_left = math.atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                suitable_V.append(new_v)
                rad_s = rad
            else:
                unsuitable_V.append(new_v) 
                rad_s = 1
    new_v = vA[:]
    suit = True
    for RVO_BA in RVO_BA_all:                
        p_0 = RVO_BA[0]
        left = RVO_BA[1]
        right = RVO_BA[2]
        dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
        theta_dif = math.atan2(dif[1], dif[0])
        theta_right = math.atan2(right[1], right[0])
        theta_left = math.atan2(left[1], left[0])
        if in_between(theta_right, theta_dif, theta_left):
            suit = False
            break
    if suit:
        suitable_V.append(new_v)
        #ang_v.append(an_v)
    else:
        unsuitable_V.append(new_v)
    #----------------------        
    if suitable_V:
        # print 'Suitable found'
        vA_post = min(suitable_V, key = lambda v: distance(v, vA))
        new_v = vA_post[:]
    else:
        # print 'Suitable not found'
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dist = RVO_BA[3]
                rad = RVO_BA[4]
                dif = [unsuit_v[0]+pA[0]-p_0[0], unsuit_v[1]+pA[1]-p_0[1]]
                theta_dif = math.atan2(dif[1], dif[0])
                theta_right = math.atan2(right[1], right[0])
                theta_left = math.atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    small_theta = abs(theta_dif-0.5*(theta_left+theta_right))
                    if abs(dist*math.sin(small_theta)) >= rad:
                        rad = abs(dist*math.sin(small_theta))
                    big_theta = math.asin(abs(dist*math.sin(small_theta))/rad)
                    dist_tg = abs(dist*math.cos(small_theta))-abs(rad*math.cos(big_theta))
                    if dist_tg < 0:
                        dist_tg = 0 
                    if (distance(dif,[0,0])>0):                   
                        tc_v = dist_tg/distance(dif, [0,0])
                    else:
                        tc_v = dist_tg
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc)+0.001
        WT = 0.2
        vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+distance(v, vA)))
        #ang_v = vA_post/rad_s
    return vA_post,rad_s

    

def RVO_update(X, V_des, V_current,rad):
    ROB_RAD = rad
    V_opt = list(V_current)    
    for i in range(len(X)):
        vA = [V_current[i][0], V_current[i][1]]
        pA = [X[i][0], X[i][1]]
        RVO_BA_all = []
        for j in range(len(X)):
            if i!=j:
                vB = [V_current[j][0], V_current[j][1]]
                pB = [X[j][0], X[j][1]]
                transl_vB_vA = [pA[0]+0.5*vB[0]+0.5*vA[0], pA[1]+0.5*vB[1]+0.5*vA[1]]
                dist_BA = distance(pA, pB)
                theta_BA = math.atan2(pB[1]-pA[1], pB[0]-pA[0])
                if 2*ROB_RAD > dist_BA:
                    dist_BA = 2*ROB_RAD
                theta_BAort = math.asin(2*ROB_RAD/dist_BA)
                theta_ort_left = theta_BA+theta_BAort
                bound_left = [math.cos(theta_ort_left), math.sin(theta_ort_left)]
                theta_ort_right = theta_BA-theta_BAort
                bound_right = [math.cos(theta_ort_right), math.sin(theta_ort_right)]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2*ROB_RAD]
                RVO_BA_all.append(RVO_BA)         
        vA_post,rad = velo_checker(pA, V_des[i], RVO_BA_all)
        V_opt[i] = vA_post
    return V_opt,rad

def avoiding_velocity_sampling(X_cord,goal,V,rad):
    V_max = np.array([1,1])
    V_des = compute_V_des(X_cord, goal, V_max)
    # print (V_des)
    V_change,rad1 = RVO_update(X_cord, V_des, V,rad)
    return V_change

def find_all_paths(X_cord,goal,nbot,rad):
    V_max = np.array([1,1])
    step = 0.1
    V = [[0,0] for i in range(nbot)]
    xarr = [[]for i in range(nbot)]
    yarr = [[] for i in range(nbot)]
    dists = []
    for i in range(nbot):
        dist = distance(X_cord[i],goal[i])
        dists.append(dist)
    dists = np.array(dists)
    while np.max(dists)>0.2:
        V_des = compute_V_des(X_cord, goal, V_max)
        V_change,rad1 = RVO_update(X_cord, V_des, V, rad)
        V = [[(V[i][0]*0.5+V_change[i][0]*0.5),(V[i][1]*0.5+V_change[i][1]*0.5)] for i in range(nbot)]
        for i in range(len(X_cord)):
            X_cord[i][0] += V[i][0]*step
            X_cord[i][1] += V[i][1]*step
            xarr[i].append(X_cord[i][0])
            yarr[i].append(X_cord[i][1])
            dists[i] = distance(X_cord[i],goal[i])
            print ("Agent ",i," with distance ",distance(X_cord[i],goal[i]))
    # xarr = np.array(xarr)
    # yarr = np.array(yarr)
    # print (xarr.shape)
    return xarr,yarr




# cur_pos = np.array([[0.0, 0.0, 0.0],[0.0, 5.0, 0.0],[5.0,0.0,0.0],[0.0,0.0,5.0]])
# nbot = 4
# V = np.array([[0.15,0.0,0.0],[0.0,0.15,0.0],[0.15,0.0,0.15],[0.0,0.0,0.15]])
# goal = np.array([[5.0, 0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,5.0],[0.0,5.0,0.0]])
# total_time = 10
# del_t = 0.1
# robot_rad = 0.2
# t = 0
# xarr = [[] for i in range(nbot)]
# yarr = [[] for i in range(nbot)]
# zarr = [[] for i in range(nbot)]

# while t*del_t < total_time:
#     vx,vy,vz = avoiding_velocity(nbot, cur_pos, V, goal, robot_rad, del_t)
#     V = [[(vx[i],vy[i],vz[i])] for i in range(len(vx))]
#     V = np.array(V)
#     print (V)
#     cmap = get_cmap(nbot)
#     for j in range(nbot):
#         cur_pos[j][0] += vx[j]*del_t
#         cur_pos[j][1] += vy[j]*del_t
#         cur_pos[j][2] += vz[j]*del_t
#         xarr[j].append(cur_pos[j][0])
#         yarr[j].append(cur_pos[j][1])
#         zarr[j].append(cur_pos[j][2])
#         plt.plot(xarr[j],yarr[j],zarr[j],color=cmap(j),marker="o",linewidth = 0.05,markersize=0.05)
#     visualize_traj_dynamic(xarr,yarr,zarr,cur_pos, V, robot_rad, goal, time=t*del_t, name='data/snap%s.png'%str(t))
#     t += 1
