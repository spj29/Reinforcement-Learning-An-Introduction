import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d
from tiles3 import IHT, tiles

x_min=-1.2
x_max=0.5
v_min=-0.07
v_max=0.07
gamma=1
actions=[-1.0, 0.0, 1.0]

def normalize_v(v):

	if v<v_min:

		return v_min

	if v>v_max:

		return v_max

	return v

def normalize_x(x):

	if x<x_min:

		return x_min

	if x>x_max:

		return x_max

	return x

def get_val(w, tilings, iht, Xscale, Vscale):

	X=np.linspace(x_min, x_max+0.1, 50)
	V=np.linspace(v_min, v_max, 50)
	x, y=np.meshgrid(X, V)
	value=np.zeros((50, 50), dtype='float')

	for i in range(50):

		for j in range(50):

			_x, _v=(x[i, j], y[i, j])

			if _x>x_max:

				value[49-i, j]=0
			
			else:

				val=[]

				for k in range(len(actions)):
					
					features=tiles(iht, tilings, [_x*Xscale, _v*Vscale], [actions[k]])

					val.append(np.sum(w[features]))

				value[49-i, j]=-np.max(val)

	return value


def semiGradientSARSA(episodes, tilings, alpha, eps, sample_ep=[], sample_step=[]):

	iht=IHT(4096)
	w=np.zeros(4096)
	steps_per_episode=np.zeros(episodes)
	total_steps=0
	weight_episode=[]
	weight_step=[]

	Xscale=tilings/(x_max-x_min)
	Vscale=tilings/(v_max-v_min)

	for episode in range(episodes):

		x=np.random.uniform(-0.6, -0.4)
		v=0
		a=np.random.randint(len(actions))

		while x!=x_max:

			steps_per_episode[episode]+=1
			total_steps+=1

			v_nxt=normalize_v(v+0.001*actions[a]-0.0025*np.cos(3*x))
			x_nxt=normalize_x(x+v_nxt)
			
			if x_nxt==x_min:

				v_nxt=0

			a_nxt=0
			val_nxt=0

			if np.random.rand()<eps:

				a_nxt=np.random.randint(len(actions))
				features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[a_nxt]])
				val_nxt=np.sum(w[features])

			else:

				val=[]

				for i in range(len(actions)):
					
					features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[i]])
					val.append(np.sum(w[features]))

				a_nxt=np.random.choice(np.ndarray.flatten(np.argwhere(val==np.max(val))))
				val_nxt=val[a_nxt]

			features=tiles(iht, tilings, [x*Xscale, v*Vscale], [actions[a]])

			w[features]=np.add(w[features], (alpha/tilings)*(-1+gamma*val_nxt-np.sum(w[features])))

			(x, v, a)=(x_nxt, v_nxt, a_nxt)
			
			if total_steps in sample_step:

				weight_step.append(get_val(w, tilings, iht, Xscale, Vscale))


		if episode+1 in sample_ep:

			weight_episode.append(get_val(w, tilings, iht, Xscale, Vscale))

	return steps_per_episode, weight_episode, weight_step

def figure_10_1():

	episodes=9000
	tilings=8
	alpha=0.5
	eps=0

	sample_ep=[12, 104, 1000, 5000, 9000]
	sample_step=[428]

	_, weight_episode, weight_step=semiGradientSARSA(episodes, tilings, alpha, eps, sample_ep, sample_step)

	X=np.linspace(x_min, x_max+0.1, 50)
	Y=np.linspace(v_min, v_max, 50)
	x, y=np.meshgrid(X, Y)
	fig=plt.figure(figsize=(15,20))
	index=1

	ax=fig.add_subplot(3, 2, index, projection='3d')
	ax.plot_wireframe(x, y, weight_step[0])
	ax.set_xlabel("Position")
	ax.set_ylabel("Velocity")
	ax.set_zlabel("Cost-To-Go")
	ax.set_title('Steps: {}'.format(sample_step[0]))
	index+=1

	for i, ep in enumerate(sample_ep):

		ax=fig.add_subplot(3, 2, index, projection='3d')
		ax.plot_wireframe(x, y, weight_episode[i])
		ax.set_xlabel("Position")
		ax.set_ylabel("Velocity")
		ax.set_zlabel("Cost-To-Go")
		ax.set_title('Episodes: {}'.format(ep))
		index+=1

	plt.savefig('figure_10_1.png')
	plt.close()

def figure_10_2():

	episodes=500
	tilings=8
	alpha=[0.1, 0.2, 0.5]
	eps=0.1
	steps=[np.zeros(episodes) for a in alpha]
	runs=100

	for run in tqdm(range(runs)):

		for i, a in enumerate(alpha):

			step, _, __=semiGradientSARSA(episodes, tilings, a, eps)
			steps[i]+=step

	for i in range(len(alpha)):

		steps[i]/=runs
		plt.plot(steps[i], label=r'$\alpha={}/{}$'.format(alpha[i], tilings))

	plt.xlabel('Episodes')
	plt.ylabel('Steps Per Episode')
	plt.yscale('log', basey=2)
	plt.title('Mountain Car Semi Gradient Sarsa')
	plt.legend()
	plt.savefig('figure_10_2.png')
	plt.close()

def Main():

	figure_10_1()
	figure_10_2()

Main()
