
from numpy import *
import matplotlib.pyplot as plt
import scipy.special
import os, sys
from scipy import interpolate
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
paths = os.getcwd()+"/"

rad = sys.argv[1]


#####################robots trajectories ###########################
x_temp = scipy.io.loadmat(paths+'x.mat')
x_1 = x_temp['x'].squeeze()

y_temp = scipy.io.loadmat(paths+'y.mat')
y_1 = y_temp['y'].squeeze()

z_temp = scipy.io.loadmat(paths+'z.mat')
z_1 = z_temp['z'].squeeze()
# z_1 = z_1*10**3



# x_temp = scipy.io.loadmat('x_2.mat')
# x_2 = x_temp['x_2'].squeeze()

# y_temp = scipy.io.loadmat('y_2.mat')
# y_2 = y_temp['y_2'].squeeze()

# z_temp = scipy.io.loadmat('z_2.mat')
# z_2 = z_temp['z_2'].squeeze()


# x_temp = scipy.io.loadmat('x_3.mat')
# x_3 = x_temp['x_3'].squeeze()

# y_temp = scipy.io.loadmat('y_3.mat')
# y_3 = y_temp['y_3'].squeeze()

# z_temp = scipy.io.loadmat('z_3.mat')
# z_3 = z_temp['z_3'].squeeze()

#################ploting ####################

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_1[0,:-1],y_1[0,:-1],z_1[0,:-1], '-r', linewidth = 3.0)
ax.plot(x_1[1,:-1],y_1[1,:-1],z_1[1,:-1], color ='pink', linewidth = 3.0)
ax.plot(x_1[2,:-1],y_1[2,:-1],z_1[2,:-1], color ='orange', linewidth = 3.0)
ax.plot(x_1[3,:-1],y_1[3,:-1],z_1[3,:-1], color ='green', linewidth = 3.0)

ax.scatter3D(x_1[0,0], y_1[0,0], z_1[0,0], s=13**2,color = "red", marker ="x") 
ax.scatter3D(x_1[1,0], y_1[1,0], z_1[1,0], s=13**2,color = "pink", marker ="x") 
ax.scatter3D(x_1[2,0], y_1[2,0], z_1[2,0], s=13**2,color = "orange", marker ="x") 
ax.scatter3D(x_1[3,0], y_1[3,0], z_1[3,0], s=13**2,color = "green", marker ="x") 

ax.scatter3D(x_1[0,-1], y_1[0,-1], z_1[0,-1],s=8**2, color = "red") 
ax.scatter3D(x_1[1,-1], y_1[1,-1], z_1[1,-1],s=8**2, color = "pink") 
ax.scatter3D(x_1[2,-1], y_1[2,-1], z_1[2,-1],s=8**2, color = "orange") 
ax.scatter3D(x_1[3,-1], y_1[3,-1], z_1[3,-1],s=8**2, color = "green") 

ax.set_xlabel('x[m]', fontweight ='bold')  
ax.set_ylabel('y[m]', fontweight ='bold')  
ax.set_zlabel('z[m]', fontweight ='bold') 
# ax.text2D(0.05, 0.95, "r=0.30", transform=ax.transAxes)



# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_2[0],y_2[0],z_2[0], '-r', linewidth = 3.0)
# ax.plot(x_2[1],y_2[1],z_2[1], color ='pink', linewidth = 3.0)
# ax.plot(x_2[2],y_2[2],z_2[2], color ='orange', linewidth = 3.0)
# ax.plot(x_2[3],y_2[3],z_2[3], color ='green', linewidth = 3.0)
# ax.plot(x_2[4],y_2[4],z_2[4], color ='grey', linewidth = 3.0)
# ax.plot(x_2[5],y_2[5],z_2[5], color ='navy', linewidth = 3.0)
# ax.plot(x_2[6],y_2[6],z_2[6], color ='yellow', linewidth = 3.0)
# ax.plot(x_2[7],y_2[7],z_2[7], color ='purple', linewidth = 3.0)
# ax.plot(x_2[8],y_2[8],z_2[8], color ='darkgreen', linewidth = 3.0)
# ax.plot(x_2[9],y_2[9],z_2[9], color ='teal', linewidth = 3.0)
# ax.plot(x_2[10],y_2[10],z_2[10], color ='salmon', linewidth = 3.0)
# ax.plot(x_2[11],y_2[11],z_2[11], color ='black', linewidth = 3.0)
# ax.plot(x_2[12],y_2[12],z_2[12], color ='cyan', linewidth = 3.0)
# ax.plot(x_2[13],y_2[13],z_2[13], color ='brown', linewidth = 3.0)
# ax.plot(x_2[14],y_2[14],z_2[14], color ='coral', linewidth = 3.0)
# ax.plot(x_2[15],y_2[15],z_2[15], color ='lime', linewidth = 3.0)
# ax.scatter3D(x_2[0,0], y_2[0,0], z_2[0,0], s=13**2,color = "red", marker ="x") 
# ax.scatter3D(x_2[1,0], y_2[1,0], z_2[1,0], s=13**2,color = "pink", marker ="x") 
# ax.scatter3D(x_2[2,0], y_2[2,0], z_2[2,0], s=13**2,color = "orange", marker ="x") 
# ax.scatter3D(x_2[3,0], y_2[3,0], z_2[3,0], s=13**2,color = "green", marker ="x") 
# ax.scatter3D(x_2[4,0], y_2[4,0], z_2[4,0], s=13**2,color = "grey", marker ="x") 
# ax.scatter3D(x_2[5,0], y_2[5,0], z_2[5,0], s=13**2,color = "navy", marker ="x") 
# ax.scatter3D(x_2[6,0], y_2[6,0], z_2[6,0], s=13**2,color = "yellow", marker ="x") 
# ax.scatter3D(x_2[7,0], y_2[7,0], z_2[7,0], s=13**2,color = "purple", marker ="x") 
# ax.scatter3D(x_2[8,0], y_2[8,0], z_2[8,0], s=13**2,color = "darkgreen", marker ="x") 
# ax.scatter3D(x_2[9,0], y_2[9,0], z_2[9,0], s=13**2,color = "teal", marker ="x") 
# ax.scatter3D(x_2[10,0], y_2[10,0], z_2[10,0],s=13**2, color = "salmon", marker ="x") 
# ax.scatter3D(x_2[11,0], y_2[11,0], z_2[11,0],s=13**2, color = "black", marker ="x") 
# ax.scatter3D(x_2[12,0], y_2[12,0], z_2[12,0],s=13**2, color = "cyan", marker ="x") 
# ax.scatter3D(x_2[13,0], y_2[13,0], z_2[13,0],s=13**2, color = "brown", marker ="x") 
# ax.scatter3D(x_2[14,0], y_2[14,0], z_2[14,0],s=13**2, color = "coral", marker ="x") 
# ax.scatter3D(x_2[15,0], y_2[15,0], z_2[15,0],s=13**2, color = "lime", marker ="x") 
# ax.scatter3D(x_2[0,-1], y_2[0,-1], z_2[0,-1],s=8**2, color = "red") 
# ax.scatter3D(x_2[1,-1], y_2[1,-1], z_2[1,-1],s=8**2, color = "pink") 
# ax.scatter3D(x_2[2,-1], y_2[2,-1], z_2[2,-1],s=8**2, color = "orange") 
# ax.scatter3D(x_2[3,-1], y_2[3,-1], z_2[3,-1],s=8**2, color = "green") 
# ax.scatter3D(x_2[4,-1], y_2[4,-1], z_2[4,-1],s=8**2, color = "grey") 
# ax.scatter3D(x_2[5,-1], y_2[5,-1], z_2[5,-1],s=8**2, color = "navy") 
# ax.scatter3D(x_2[6,-1], y_2[6,-1], z_2[6,-1],s=8**2, color = "yellow") 
# ax.scatter3D(x_2[7,-1], y_2[7,-1], z_2[7,-1],s=8**2, color = "purple") 
# ax.scatter3D(x_2[8,-1], y_2[8,-1], z_2[8,-1],s=8**2, color = "darkgreen") 
# ax.scatter3D(x_2[9,-1], y_2[9,-1], z_2[9,-1],s=8**2, color = "teal") 
# ax.scatter3D(x_2[10,-1], y_2[10,-1], z_2[10,-1],s=8**2, color = "salmon") 
# ax.scatter3D(x_2[11,-1], y_2[11,-1], z_2[11,-1],s=8**2, color = "black") 
# ax.scatter3D(x_2[12,-1], y_2[12,-1], z_2[12,-1],s=8**2, color = "cyan") 
# ax.scatter3D(x_2[13,-1], y_2[13,-1], z_2[13,-1],s=8**2, color = "brown") 
# ax.scatter3D(x_2[14,-1], y_2[14,-1], z_2[14,-1],s=8**2, color = "coral") 
# ax.scatter3D(x_2[15,-1], y_2[15,-1], z_2[15,-1],s=8**2, color = "lime") 
# ax.set_xlabel('x[m]', fontweight ='bold')  
# ax.set_ylabel('y[m]', fontweight ='bold')  
# ax.set_zlabel('z[m]', fontweight ='bold') 
# ax.text2D(0.05, 0.95, "r=0.25", transform=ax.transAxes)



# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_3[0],y_3[0],z_3[0], '-r', linewidth = 3.0)
# ax.plot(x_3[1],y_3[1],z_3[1], color ='pink', linewidth = 3.0)
# ax.plot(x_3[2],y_3[2],z_3[2], color ='orange', linewidth = 3.0)
# ax.plot(x_3[3],y_3[3],z_3[3], color ='green', linewidth = 3.0)
# ax.plot(x_3[4],y_3[4],z_3[4], color ='grey', linewidth = 3.0)
# ax.plot(x_3[5],y_3[5],z_3[5], color ='navy', linewidth = 3.0)
# ax.plot(x_3[6],y_3[6],z_3[6], color ='yellow', linewidth = 3.0)
# ax.plot(x_3[7],y_3[7],z_3[7], color ='purple', linewidth = 3.0)
# ax.plot(x_3[8],y_3[8],z_3[8], color ='darkgreen', linewidth = 3.0)
# ax.plot(x_3[9],y_3[9],z_3[9], color ='teal', linewidth = 3.0)
# ax.plot(x_3[10],y_3[10],z_3[10], color ='salmon', linewidth = 3.0)
# ax.plot(x_3[11],y_3[11],z_3[11], color ='black', linewidth = 3.0)
# ax.plot(x_3[12],y_3[12],z_3[12], color ='cyan', linewidth = 3.0)
# ax.plot(x_3[13],y_3[13],z_3[13], color ='brown', linewidth = 3.0)
# ax.plot(x_3[14],y_3[14],z_3[14], color ='coral', linewidth = 3.0)
# ax.plot(x_3[15],y_3[15],z_3[15], color ='lime', linewidth = 3.0)
# ax.scatter3D(x_3[0,0], y_3[0,0], z_3[0,0], s=13**2,color = "red", marker ="x") 
# ax.scatter3D(x_3[1,0], y_3[1,0], z_3[1,0], s=13**2,color = "pink", marker ="x") 
# ax.scatter3D(x_3[2,0], y_3[2,0], z_3[2,0], s=13**2,color = "orange", marker ="x") 
# ax.scatter3D(x_3[3,0], y_3[3,0], z_3[3,0], s=13**2,color = "green", marker ="x") 
# ax.scatter3D(x_3[4,0], y_3[4,0], z_3[4,0], s=13**2,color = "grey", marker ="x") 
# ax.scatter3D(x_3[5,0], y_3[5,0], z_3[5,0], s=13**2,color = "navy", marker ="x") 
# ax.scatter3D(x_3[6,0], y_3[6,0], z_3[6,0], s=13**2,color = "yellow", marker ="x") 
# ax.scatter3D(x_3[7,0], y_3[7,0], z_3[7,0], s=13**2,color = "purple", marker ="x") 
# ax.scatter3D(x_3[8,0], y_3[8,0], z_3[8,0], s=13**2,color = "darkgreen", marker ="x") 
# ax.scatter3D(x_3[9,0], y_3[9,0], z_3[9,0], s=13**2,color = "teal", marker ="x") 
# ax.scatter3D(x_3[10,0], y_3[10,0], z_3[10,0],s=13**2, color = "salmon", marker ="x") 
# ax.scatter3D(x_3[11,0], y_3[11,0], z_3[11,0],s=13**2, color = "black", marker ="x") 
# ax.scatter3D(x_3[12,0], y_3[12,0], z_3[12,0],s=13**2, color = "cyan", marker ="x") 
# ax.scatter3D(x_3[13,0], y_3[13,0], z_3[13,0],s=13**2, color = "brown", marker ="x") 
# ax.scatter3D(x_3[14,0], y_3[14,0], z_3[14,0],s=13**2, color = "coral", marker ="x") 
# ax.scatter3D(x_3[15,0], y_3[15,0], z_3[15,0],s=13**2, color = "lime", marker ="x") 
# ax.scatter3D(x_3[0,-1], y_3[0,-1], z_3[0,-1],s=8**2, color = "red") 
# ax.scatter3D(x_3[1,-1], y_3[1,-1], z_3[1,-1],s=8**2, color = "pink") 
# ax.scatter3D(x_3[2,-1], y_3[2,-1], z_3[2,-1],s=8**2, color = "orange") 
# ax.scatter3D(x_3[3,-1], y_3[3,-1], z_3[3,-1],s=8**2, color = "green") 
# ax.scatter3D(x_3[4,-1], y_3[4,-1], z_3[4,-1],s=8**2, color = "grey") 
# ax.scatter3D(x_3[5,-1], y_3[5,-1], z_3[5,-1],s=8**2, color = "navy") 
# ax.scatter3D(x_3[6,-1], y_3[6,-1], z_3[6,-1],s=8**2, color = "yellow") 
# ax.scatter3D(x_3[7,-1], y_3[7,-1], z_3[7,-1],s=8**2, color = "purple") 
# ax.scatter3D(x_3[8,-1], y_3[8,-1], z_3[8,-1],s=8**2, color = "darkgreen") 
# ax.scatter3D(x_3[9,-1], y_3[9,-1], z_3[9,-1],s=8**2, color = "teal") 
# ax.scatter3D(x_3[10,-1], y_3[10,-1], z_3[10,-1],s=8**2, color = "salmon") 
# ax.scatter3D(x_3[11,-1], y_3[11,-1], z_3[11,-1],s=8**2, color = "black") 
# ax.scatter3D(x_3[12,-1], y_3[12,-1], z_3[12,-1],s=8**2, color = "cyan") 
# ax.scatter3D(x_3[13,-1], y_3[13,-1], z_3[13,-1],s=8**2, color = "brown") 
# ax.scatter3D(x_3[14,-1], y_3[14,-1], z_3[14,-1],s=8**2, color = "coral") 
# ax.scatter3D(x_3[15,-1], y_3[15,-1], z_3[15,-1],s=8**2, color = "lime") 
# ax.set_xlabel('x[m]', fontweight ='bold')  
# ax.set_ylabel('y[m]', fontweight ='bold')  
# ax.set_zlabel('z[m]', fontweight ='bold') 
# ax.text2D(0.05, 0.95, "r=0.15", transform=ax.transAxes)

plt.savefig("results/4_agents_"+str(rad)+".jpg")
plt.show()

