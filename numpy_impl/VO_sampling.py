import numpy as np
import math

def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return math.sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2+(pose1[2]-pose2[2])**2)+0.001

def reach(p1, p2, bound=0.5):
    if distance(p1,p2)< bound:
        return True
    else:
        return False

def compute_V_des(X, goal, V_max):
    V_des = []
    nbot = len(X)
    print (nbot)
    for i in range(len(X)):
        dif_x = [goal[i][k]-X[i][k] for k in range(3)]
        norm = distance(dif_x, [0, 0, 0])
        norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(3)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.1):
            V_des[i][0] = 0
            V_des[i][1] = 0
            V_des[i][2] = 0
    return V_des
            

def VO_update(X, V_des, V_current, ws_model,robrad,obst):
    ROB_RAD = robrad
    V_opt = list(V_current)    
    for i in range(len(X)):
        vA = [V_current[i][0], V_current[i][1], V_current[i][2]]
        pA = [X[i][0], X[i][1], X[i][2]]
        RVO_BA_all = []
        for j in range(len(X)):
            if i!=j:
                vB = [V_current[j][0], V_current[j][1], V_current[j][2]]
                pB = [X[j][0], X[j][1], X[j][2]]
                transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1], pA[2]+vB[2]]
                dist_BA = distance(pA, pB)
                theta_BA = get_angles(pB-pA)
                if 2*ROB_RAD > dist_BA:
                    dist_BA = 2*ROB_RAD
                theta_BAort = math.asin(2*ROB_RAD/dist_BA)
                theta_ort_left = theta_BA+theta_BAort
                bound_left = [math.cos(theta_ort_left), math.sin(theta_ort_left)]
                theta_ort_right = theta_BA-theta_BAort
                bound_right = [math.cos(theta_ort_right), math.sin(theta_ort_right)]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2*ROB_RAD]
                RVO_BA_all.append(RVO_BA)     
        vA_post,rad = velo_checker(pA, V_des[j], VO_BA_all)
        V_opt[j] = vA_post
    return V_opt,rad

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

def get_angles(velocity):
    norm_velo = np.linalg.norm(velocity)
    dotp = np.dot(velocity,np.array([0,0,1]))
    sin_theta = dotp/norm_velo
    theta = np.arcsin(sin_theta)
    return theta

def velo_checker(pA, vA, VO_BA_all):
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    ang_v = []
    unsuitable_V = []
    rad_s = 0
    for alpha in np.arange(0, 2*math.pi, 0.1):
        for beta in np.arange(0, 2*math.pi, 0.1):
            for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
                new_v = [rad*math.sin(beta)*math.cos(alpha), rad*math.sin(beta)*math.sin(alpha), rad*math.cos(beta)]
                suit = True
                for VO_BA in VO_BA_all:
                    p_0 = VO_BA[0]
                    left = VO_BA[1]
                    right = VO_BA[2]
                    dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1],new_v[2]+pA[2]-p_0[2]]
                    theta_dif = get_angles(dif)
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
    for VO_BA in VO_BA_all:                
        p_0 = VO_BA[0]
        left = VO_BA[1]
        right = VO_BA[2]
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
            for VO_BA in VO_BA_all:
                p_0 = VO_BA[0]
                left = VO_BA[1]
                right = VO_BA[2]
                dist = VO_BA[3]
                rad = VO_BA[4]
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
                    tc_v = dist_tg/distance(dif, [0,0])
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc)+0.001
        WT = 0.2
        vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+distance(v, vA)))
        #ang_v = vA_post/rad_s
    return vA_post,rad_s

    