import math
import numpy as np
import json

R = 25
centreX = 12
centreY = 7
n = 65
theta = np.linspace(0,2*math.pi,n)
x = centreX+R*np.cos(theta)
y = centreY+R*np.sin(theta)

start_end = []
for i in range(n-1):
    d = {}
    rot = 12
    start_pos  = [x[i],y[i],1]
    end_pos = [x[(i+rot)%(n-1)],y[(i+rot)%(n-1)],1]
    d["name"] = "newbot"
    d["start"] = start_pos
    d["goal"] = end_pos
    d["radius"] = 0.12
    # d["v_init"] = [0.0,0.0,0.0]
    # d["v_fin"] = [0.0,0.0,0.0]
    # d["a_init"] = [0.0,0.0,0.0]
    # d["a_fin"] = [0.0,0.0,0.0]
    d["speed"] = 1.0
    start_end.append(d)

agents = {"agents":start_end}
json_obj = json.dumps(agents,indent = 1)
with open("dataset_rviz/"+str(n-1)+"_agents_circle.json","w") as out:
    out.write(json_obj)


