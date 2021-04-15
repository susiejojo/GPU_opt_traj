import numpy as np
import json

def init_final_pos():

	f = open("dataset/8_agents_12.json","r")
	data = json.loads(f.read())
	x_init = []
	y_init = []
	z_init = []
	x_fin = []
	y_fin = []
	z_fin = []
	nbot = len(data["agents"])
	rad = data["agents"][0]["radius"]
	for i in data["agents"]:
		x_init.append(i["start"][0])
		y_init.append(i["start"][1])
		z_init.append(i["start"][2])
		x_fin.append(i["goal"][0])
		y_fin.append(i["goal"][1])
		z_fin.append(i["goal"][2])

	x_init = np.array(x_init)
	y_init = np.array(y_init)
	z_init = np.array(z_init)

	x_fin = np.array(x_fin)
	y_fin = np.array(y_fin)
	z_fin = np.array(z_fin)

	# x_fin = hstack(( x_fin_1, x_fin_2, x_fin_3, x_fin_4, x_fin_5, x_fin_6, x_fin_7, x_fin_8, x_fin_9, x_fin_10, x_fin_11, x_fin_12, x_fin_13, x_fin_14, x_fin_15, x_fin_16))
	# y_fin = hstack(( y_fin_1, y_fin_2, y_fin_3, y_fin_4, y_fin_5, y_fin_6, y_fin_7, y_fin_8, y_fin_9, y_fin_10, y_fin_11, y_fin_12, y_fin_13, y_fin_14, y_fin_15, y_fin_16))
	# z_fin = hstack(( z_fin_1, z_fin_2, z_fin_3, z_fin_4, z_fin_5, z_fin_6, z_fin_7, z_fin_8, z_fin_9, z_fin_10, z_fin_11, z_fin_12, z_fin_13, z_fin_14, z_fin_15, z_fin_16))

	return x_init, y_init, z_init, x_fin, y_fin, z_fin, rad, nbot

# print (init_final_pos())

