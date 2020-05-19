import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d
from tiles3 import IHT, tiles
from collections import deque

x_min=-1.2
x_max=0.5
v_min=-0.07
v_max=0.07
gamma=1
actions=[-1.0, 0.0, 1.0]
limit=5000

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

def SARSALambda(lamda, episodes, tilings, alpha, eps):

	iht=IHT(4096)
	w=np.zeros(4096)
	steps_per_episode=np.zeros(episodes)

	Xscale=tilings/(x_max-x_min)
	Vscale=tilings/(v_max-v_min)

	for episode in range(episodes):

		z=np.zeros(4096)
		x=np.random.uniform(-0.6, -0.4)
		v=0
		a=np.random.randint(len(actions))
		T=0

		while x!=x_max:

			T+=1
			steps_per_episode[episode]+=1

			if(steps_per_episode[episode]>limit):

				print('Step Limit Exceeded')
				break

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
			delta=-1+gamma*val_nxt-np.sum(w[features])

			z=gamma*lamda*z
			z[features]=1

			w=np.add(w, (alpha/tilings)*delta*z)

			(x, v, a)=(x_nxt, v_nxt, a_nxt)
	
	return steps_per_episode

def figure_12_10():

	episodes=50
	tilings=8
	N=[0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]
	alpha=[np.arange(0.5, 1.8, 0.1), np.arange(0.4, 1.8, 0.1), np.arange(0.3, 1.8, 0.1), np.arange(0.3, 1.8, 0.1), np.arange(0.3, 1.8, 0.1), np.arange(0.3, 1.6, 0.1), np.arange(0.3, 1.5, 0.1)]
	eps=0
	steps=[np.zeros(len(alpha[i])) for i in range(len(N))]
	runs=30

	for run in tqdm(range(runs)):

		for i, lamda in enumerate(N):

			for j, a in enumerate(alpha[i]):

				steps[i][j]+=np.mean(SARSALambda(lamda, episodes, tilings, np.round(a, 2), eps))
				
				print(lamda ,np.round(a, 2))

	for i in range(len(N)):

		steps[i]/=runs
		plt.plot(alpha[i], steps[i], label=r'$\lambda={}$'.format(N[i]))

	plt.xlabel(r'$\alpha*number of tilings({})$'.format(tilings))
	plt.ylabel('Steps Per Episode Averaged over first 50 episodes')
	plt.title(r'$Mountain \ Car \ Sarsa(\lambda)$')
	plt.legend()
	plt.savefig('figure_12_10.png')
	plt.close()

def Main():

	figure_12_10()

Main()
