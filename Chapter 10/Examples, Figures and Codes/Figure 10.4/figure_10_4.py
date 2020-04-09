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

def semiGradientNStepSARSA(n, episodes, tilings, alpha, eps):

	iht=IHT(4096)
	w=np.zeros(4096)
	steps_per_episode=np.zeros(episodes)

	Xscale=tilings/(x_max-x_min)
	Vscale=tilings/(v_max-v_min)

	for episode in range(episodes):

		trajectory=deque()
		x=np.random.uniform(-0.6, -0.4)
		v=0
		a=np.random.randint(len(actions))
		T=0

		trajectory.append([x, v, a , -1])

		while x!=x_max:

			T+=1
			steps_per_episode[episode]+=1

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


			t=T-n

			if t>=0:

				total_return=-len(trajectory)
				total_return+=val_nxt

				_x, _v, _a, _r=trajectory.popleft()
				
				features=tiles(iht, tilings, [_x*Xscale, _v*Vscale], [actions[_a]])
				w[features]=np.add(w[features], (alpha/tilings)*(total_return-np.sum(w[features])))

			trajectory.append([x_nxt, v_nxt, a_nxt, -1])
			(x, v, a)=(x_nxt, v_nxt, a_nxt)

		trajectory.pop()

		while len(trajectory):

			total_return=-len(trajectory)

			_x, _v, _a, _r=trajectory.popleft()
			
			features=tiles(iht, tilings, [_x*Xscale, _v*Vscale], [actions[_a]])
			w[features]=np.add(w[features], (alpha/tilings)*(total_return-np.sum(w[features])))
	
	return steps_per_episode

def figure_10_4():

	episodes=50
	tilings=8
	N=np.power(2, np.arange(0, 5))
	alpha=[np.arange(0.5, 1.8, 0.1), np.arange(0.3, 1.8, 0.1), np.arange(0.3, 1.6, 0.1), np.arange(0.1, 1.1, 0.1), np.arange(0.1, 0.7, 0.1)]
	eps=0
	steps=[np.zeros(len(alpha[i])) for i in range(len(N))]
	runs=30

	for run in tqdm(range(runs)):

		for i, n in enumerate(N):

			for j, a in enumerate(alpha[i]):

				steps[i][j]+=np.mean(semiGradientNStepSARSA(n, episodes, tilings, np.round(a, 2), eps))
				
				print(n,np.round(a, 2))

	for i in range(len(N)):

		steps[i]/=runs
		plt.plot(alpha[i], steps[i], label='n={}'.format(N[i]))

	plt.xlabel(r'$\alpha*number of tilings({})$'.format(tilings))
	plt.ylabel('Steps Per Episode Averaged over first 50 episodes')
	plt.title('Mountain Car Semi Gradient N-Step Sarsa')
	plt.legend()
	plt.savefig('___figure_10_4.png')
	plt.close()

def Main():

	figure_10_4()

Main()
