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

def SARSALambda(lamda, episodes, tilings, alpha, eps, trace='Replacing'):

	iht=IHT(4096)
	w=np.zeros(4096)
	steps_per_episode=np.zeros(episodes)

	Xscale=tilings/(x_max-x_min)
	Vscale=tilings/(v_max-v_min)

	for episode in range(episodes):

		z=np.zeros(4096)
		x=np.random.uniform(-0.6, -0.4)
		v=0
		val=[]
		for i in range(len(actions)):

			features=tiles(iht, tilings, [x*Xscale, v*Vscale], [actions[i]])
			val.append(np.sum(w[features]))

		a=np.argmax(val);
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

			if np.random.binomial(1, eps)>0:

				a_nxt=np.random.randint(len(actions))
				features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[a_nxt]])
				val_nxt=np.sum(w[features])

			else:

				val=[]

				for i in range(len(actions)):
					
					features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[i]])
					val.append(np.sum(w[features]))

				a_nxt=np.argmax(val);
				val_nxt=val[a_nxt]

			features=tiles(iht, tilings, [x*Xscale, v*Vscale], [actions[a]])
			delta=-1+gamma*val_nxt-np.sum(w[features])

			z=gamma*lamda*z

			if trace=='Replacing':
				
				z[features]=1

			elif trace=='Replacing Clearing':

				for ca in actions:

					if ca!=actions[a]:

						clear_features=tiles(iht, tilings, [x*Xscale, v*Vscale], [ca])
						z[clear_features]=0

				z[features]=1

			elif trace=='Accumulating':

				z[features]+=1

			w=np.add(w, (alpha/tilings)*delta*z)

			(x, v, a)=(x_nxt, v_nxt, a_nxt)
	
	return steps_per_episode

def trueOnlineSARSA(lamda, episodes, tilings, alpha, eps):

	iht=IHT(4096)
	w=np.zeros(4096)
	steps_per_episode=np.zeros(episodes)

	Xscale=tilings/(x_max-x_min)
	Vscale=tilings/(v_max-v_min)

	for episode in range(episodes):

		z=np.zeros(4096)
		Qold=0
		x=np.random.uniform(-0.6, -0.4)
		v=0
		val=[]
		for i in range(len(actions)):

			features=tiles(iht, tilings, [x*Xscale, v*Vscale], [actions[i]])
			val.append(np.sum(w[features]))

		a=np.argmax(val);
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

			if np.random.binomial(1, eps)>0:

				a_nxt=np.random.randint(len(actions))
				features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[a_nxt]])
				val_nxt=np.sum(w[features])

			else:

				val=[]

				for i in range(len(actions)):
					
					features=tiles(iht, tilings, [x_nxt*Xscale, v_nxt*Vscale], [actions[i]])
					val.append(np.sum(w[features]))

				a_nxt=np.argmax(val);
				val_nxt=val[a_nxt]

			features=tiles(iht, tilings, [x*Xscale, v*Vscale], [actions[a]])
			val=np.sum(w[features])
			delta=-1+gamma*val_nxt-val
			sum_z=np.sum(z[features])

			z=gamma*lamda*z
			z[features]+=1-(alpha/tilings)*gamma*lamda*sum_z

			# Using the True Online update with dutch trace Does not work at high Lambda.(I am not sure why)
			# w=np.add(w, (alpha/tilings)*(delta+val-Qold)*z)
			# w[features]-=(alpha/tilings)*(val-Qold)

			w=np.add(w, (alpha/tilings)*delta*z)

			Qold=val_nxt

			(x, v, a)=(x_nxt, v_nxt, a_nxt)
	
	return steps_per_episode

def figure_12_11():

	episodes=20
	tilings=8
	lamda=0.9
	
	name=[r'$Sarsa(\lambda) \ with \ Replacing \ Trace$',\
	r'$Sarsa(\lambda) \ with \ Replacing \ and \ Clearing \ Trace$',\
	r'$Sarsa(\lambda) \ with \ Accumulating \ Trace$']

	trace=['Replacing', 'Replacing Clearing', 'Accumulating']

	alpha=np.arange(0.2, 2.2, 0.2)
	eps=0
	rewards_true=np.zeros(len(alpha))
	rewards=[np.zeros(len(alpha)) for i in range(len(name))]
	runs=100

	def Plot(run, file_name):

		plt.figure(figsize=(8,8))

		for i in range(len(trace)):

			plt.plot(alpha, rewards[i]/run, label=name[i])

		plt.plot(alpha, rewards_true/run, label=r'$True \ Online \ Sarsa(\lambda)$')
		
		plt.xlabel(r'$\alpha*number of tilings({})$'.format(tilings))
		plt.ylim(-550,-150)
		plt.xlim(0.2, 2)
		plt.xticks(np.arange(0.2, 2.2, 0.2))
		plt.ylabel('Reward Per Episode Averaged over first {} episodes'.format(episodes))
		plt.title('Mountain Car')
		plt.legend(loc="lower right")
		plt.savefig(file_name)
		plt.close()

	for run in tqdm(range(runs)):

		for j, a in enumerate(alpha):

			for i,tr in enumerate(trace):

				if tr=='Accumulating' and a>0.4:

					rewards[i][j]-=limit

				else:
				
					rewards[i][j]-=np.mean(SARSALambda(lamda, episodes, tilings, np.round(a, 2), eps, tr))
				
				print(tr ,np.round(a, 2))

			rewards_true[j]-=np.mean(trueOnlineSARSA(lamda, episodes, tilings, np.round(a, 2), eps))

		if (run+1)%100==0:

			Plot(run+1, 'figure_12_11_{}.png'.format(run+1))


def Main():

	figure_12_11()

Main()
