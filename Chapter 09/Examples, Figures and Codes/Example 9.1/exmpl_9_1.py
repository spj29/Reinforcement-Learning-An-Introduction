import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha=2e-5

class RandomWalk:

	def __init__(self, _n):

		self.n=_n
		self.value=np.zeros(_n+1, dtype='float')

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

	def dynamicProgrammingTrueValue(self, error=1e-2):

		self.values=np.random.rand()*2-np.ones(self.n+1)

		while True:

			total_delta=0

			for state in range(1, self.n+1):

				delta=-self.value[state]

				for sign in [-1, 1]:

					for action in range(1,101):

						new=state+action*sign
						
						reward=0
						val=0

						if new>self.n:

							reward=1

						elif new<=0:

							reward=-1

						else:

							val=self.value[new]

						delta+=(reward+val)/200

				total_delta+=np.abs(delta)		
				self.value[state]+=delta

			print('Total Delta: {}'.format(total_delta))

			if total_delta<error:

				break

	def monteCarloTrueValue(self, episodes):

		for start_state in tqdm(range(1, self.n+1)):

			for i in range(episodes):

				episode, reward=self.generateEpisode(start_state)

				for state in episode:

					self.value[state]+=alpha*(reward-self.value[state])

	def monteCarloFunctionApproximation(self, episodes):

		weight=np.zeros(10, dtype='float')

		for i in tqdm(range(episodes)):

			episode, reward=self.generateEpisode()

			for state in episode:

				group=(state-1)//(self.n//10)
				weight[group]+=alpha*(reward-weight[group])
				self.value[state]=weight[group]

def Main():

	global alpha

	states=1000
	episodes=100000
	runs=1

	ValueMonteCarlo=np.zeros(states+1)
	ValueMonteCarloFunctionApproximation=np.zeros(states+1)

	monteCarloTrueValue=RandomWalk(states)
	monteCarloTrueValue.dynamicProgrammingTrueValue()
	ValueMonteCarlo+=monteCarloTrueValue.value

	alpha=2e-5
	monteCarloFunctionApproximation=RandomWalk(states)
	monteCarloFunctionApproximation.monteCarloFunctionApproximation(episodes)
	ValueMonteCarloFunctionApproximation+=monteCarloFunctionApproximation.value


	plt.plot(ValueMonteCarlo, label='True Values')
	plt.plot(ValueMonteCarloFunctionApproximation, label='Values Using State Aggregation')

	plt.xlabel('State')
	plt.ylabel('Value')
	plt.title('Monte Carlo Function Approximation')

	plt.xlim(1,1000)
	plt.ylim(-1,1)
	plt.legend()

	plt.savefig('example_9_1_DP.png')
	plt.close()

Main()