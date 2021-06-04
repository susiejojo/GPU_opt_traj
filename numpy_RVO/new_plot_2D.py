
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

rad = int(sys.argv[1])


#####################robots trajectories ###########################
x_temp = scipy.io.loadmat(paths+'x.mat')
x_1 = x_temp['x'].squeeze()

y_temp = scipy.io.loadmat(paths+'y.mat')
y_1 = y_temp['y'].squeeze()

# z_temp = scipy.io.loadmat(paths+'z.mat')
# z_1 = z_temp['z'].squeeze()
# z_1 = z_1*10**3
cmaps = {16:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime"],
        32:["red","pink","orange","green","grey","navy","yellow","purple","darkgreen","teal","salmon","black","cyan","brown","coral","lime",
            "khaki","deeppink","darkorange","crimson","darkgrey","tomato","plum","hotpink","limegreen","peru","olive","wheat","blue","orchid","gold","limegreen"]}

def visualize_traj_dynamic(x_1,y_1, time = None, name=None):
    nbot = 32
    nsamples = x_1.shape[1]
    cmap = cmaps[nbot]
    for i in range(nsamples): 
        figure = plt.figure(figsize=plt.figaspect(1))
        ax = figure.add_subplot(111) 
        for j in range(0,nbot):
            rad_corr = rad*0.01
            # ax.scatter(x_1[j,i], y_1[j,i],s=rad_corr**2,color = cmap[j])
            plt.plot(x_1[j,:i+1],y_1[j,:i+1],color = cmap[j],linewidth=1,markersize=5)    
            srec = matplotlib.patches.Circle(
                (x_1[j,i], y_1[j,i]),radius = rad_corr,
                facecolor= cmap[j],
                fill = True)
            ax.add_patch(srec)
            ax.scatter(x_1[j,0], y_1[j,0], s=10**2,color = cmap[j], marker ="x") 
            # ax.scatter(x_1[j,-1], y_1[j,-1], s=8**2, color = cmap[j])           
        ax.set_xlabel('x[m]', fontweight ='bold')  
        ax.set_ylabel('y[m]', fontweight ='bold')  
        name='data/snap%s.png'%str(i)
        plt.savefig(name, dpi = 200)
        plt.cla()
        plt.close(figure)
    return figure

visualize_traj_dynamic(x_1, y_1)
# ax.plot(x_1[0,:-1],y_1[0,:-1], '-r', linewidth = 3.0)
# ax.plot(x_1[1,:-1],y_1[1,:-1], color ='pink', linewidth = 3.0)
# ax.plot(x_1[2,:-1],y_1[2,:-1], color ='orange', linewidth = 3.0)
# ax.plot(x_1[3,:-1],y_1[3,:-1], color ='green', linewidth = 3.0)
# ax.plot(x_1[4,:-1],y_1[4,:-1], color ='grey', linewidth = 3.0)
# ax.plot(x_1[5,:-1],y_1[5,:-1], color ='navy', linewidth = 3.0)
# ax.plot(x_1[6,:-1],y_1[6,:-1], color ='yellow', linewidth = 3.0)
# ax.plot(x_1[7,:-1],y_1[7,:-1], color ='purple', linewidth = 3.0)
# ax.plot(x_1[8,:-1],y_1[8,:-1], color ='darkgreen', linewidth = 3.0)
# ax.plot(x_1[9,:-1],y_1[9,:-1], color ='teal', linewidth = 3.0)
# ax.plot(x_1[10,:-1],y_1[10,:-1], color ='salmon', linewidth = 3.0)
# ax.plot(x_1[11,:-1],y_1[11,:-1], color ='black', linewidth = 3.0)
# ax.plot(x_1[12,:-1],y_1[12,:-1], color ='cyan', linewidth = 3.0)
# ax.plot(x_1[13,:-1],y_1[13,:-1], color ='brown', linewidth = 3.0)
# ax.plot(x_1[14,:-1],y_1[14,:-1], color ='coral', linewidth = 3.0)
# ax.plot(x_1[15,:-1],y_1[15,:-1], color ='lime', linewidth = 3.0)


# ax.set_zlabel('z[m]', fontweight ='bold') 
# ax.text2D(0.05, 0.95, "r=0.30", transform=ax.transAxes)


