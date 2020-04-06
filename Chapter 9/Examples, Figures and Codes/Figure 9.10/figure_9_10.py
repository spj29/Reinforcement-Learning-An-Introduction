import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RandomWalk:

	def __init__(self, _n):

		self.n=_n

	def generateEpisode(self, start=500):

		episode=[]

		current=start
		reward=0

		while current>0 and current<self.n+1:

			action=np.random.randint(1,101)
			
			if np.random.rand()<=0.5:

				action*=-1

			new=current+action
			

			if new<=0:
				
				reward=-1

			elif new>self.n:

				reward=1

			episode.append(current)
			current=new

		return episode, reward

def dynamicProgrammingTrueValue(walk, error=1e-2):

	value=np.random.rand()*2-np.ones(walk.n+1)

	while True:

		total_delta=0

		for state in range(1, walk.n+1):

			delta=-value[state]

			for sign in [-1, 1]:

				for action in range(1,101):

					new=state+action*sign
					
					reward=0
					val=0

					if new>walk.n:

						reward=1

					elif new<=0:

						reward=-1

					else:

						val=value[new]

					delta+=(reward+val)/200

			total_delta+=np.abs(delta)		
			value[state]+=delta

		print('Total Delta: {}'.format(total_delta))

		if total_delta<error:

			break

	return value

def getFeatureVector(state, n, width, tilings):
	
	x=np.zeros((n//width+1)*tilings, dtype='float')

	for t in range(tilings):

		group=(state-1-4*t)//width+1
		x[t*((n//width)+1)+group]=1

	return x

def monteCarloMultipleTiling(walk, episodes, width, tilings, true_values):

	w=np.zeros((walk.n//width+1)*tilings, dtype='float')
	x=[getFeatureVector(i, walk.n, width, tilings) for i in range(1, walk.n+1)]
	alpha=0.0001/tilings
	errors=np.zeros(episodes)
	total=0
	mu={}

	for i in tqdm(range(episodes)):

		episode, reward=walk.generateEpisode()

		for state in episode:

			w=np.add(w, alpha*(reward-np.dot(w, x[state-1]))*x[state-1])
			total+=1
			
			prev=0
			if state in mu.keys():

				prev=mu[state]
			
			mu[state]=1+prev

		for state in range(1, walk.n+1):

			value=np.dot(w, x[state-1])
			dist=0

			if state in mu.keys():

				dist=mu[state]

			errors[i]+=(dist/total)*np.power(value-true_values[state], 2)

	errors=np.sqrt(errors)

	return errors

def Main():

	walk=RandomWalk(1000)
	runs=10
	width=200
	episodes=5000
	true_values=dynamicProgrammingTrueValue(walk)

	errors_single_tiling=np.zeros(episodes)
	errors_50_tilings=np.zeros(episodes)

	for run in range(runs):

		print('Run: {}'.format(run+1))

		errors_single_tiling+=monteCarloMultipleTiling(walk, episodes, width, 1, true_values)
		errors_50_tilings+=monteCarloMultipleTiling(walk, episodes, width, 50, true_values)

	errors_single_tiling/=runs
	errors_50_tilings/=runs

	plt.plot(errors_single_tiling, label='Single Tiling')
	plt.plot(errors_50_tilings, label='50 Tilings')

	plt.title('State Aggregation vs Tile Coding')
	plt.xlabel('Episodes')
	plt.ylabel('RMS VE averaged over {} Runs'.format(runs))
	plt.ylim(0, 0.6)
	plt.legend()
	plt.savefig('figure_9_10.png')
	plt.close()

Main()