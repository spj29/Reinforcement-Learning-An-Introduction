import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha=2e-5
gamma=1

class RandomWalk:

	def __init__(self, _n):

		self.n=_n
		self.value=np.zeros(_n+1, dtype='float')

	def generateEpisode(self, start=500, groups=20):

		episode=[]

		current=start
		reward=0

		while current>0 and current<self.n+1:

			action=np.random.randint(1,(self.n//groups)+1)
			
			if np.random.rand()<=0.5:

				action*=-1

			new=current+action
			

			if new<=0:
				
				reward=-1

			elif new>self.n:

				reward=1

			else:

				reward=0

			episode.append((current, reward))
			current=new

		episode.append((current, 0))
		return episode

	def dynamicProgrammingTrueValue(self, error=1e-2, groups=20):

		self.values=np.random.rand()*2-np.ones(self.n+1)

		while True:

			total_delta=0

			for state in range(1, self.n+1):

				delta=-self.value[state]

				for sign in [-1, 1]:

					for action in range(1,(self.n//groups)+1):

						new=state+action*sign
						
						reward=0
						val=0

						if new>self.n:

							reward=1

						elif new<=0:

							reward=-1

						else:

							val=self.value[new]

						delta+=(reward+val)/(2*(self.n//groups))

				total_delta+=np.abs(delta)		
				self.value[state]+=delta

			print('Total Delta: {}'.format(total_delta))

			if total_delta<error:

				break

	def TDZeroFunctionApproximation(self, episodes, groups=10):

		weight=np.zeros(groups, dtype='float')

		for i in tqdm(range(episodes)):

			current=self.n//2

			while current>0 and current<self.n+1:

				action=np.random.randint(1,(self.n//groups)+1)
			
				if np.random.rand()<=0.5:

					action*=-1

				new=current+action

				reward=0
				val=0

				if new>self.n:

					reward=1

				elif new<=0:

					reward=-1

				else:

					val=weight[(new-1)//(self.n//groups)]

				group=(current-1)//(self.n//groups)

				weight[group]+=alpha*(reward+gamma*val-weight[group])

				current=new			

		for state in range(1, self.n+1):

			self.value[state]=weight[(state-1)//(self.n//groups)]

	def NStepTDFunctionApproximation(self, episodes, trueValue, groups=10, n=1, alpha=0.1):

		weight=np.zeros(groups, dtype='float')
		error=0

		for episode in episodes:

			T=len(episode)
			truncatedReturn=0
			gammaPower=1

			for i in range(min(n, T-1)):	

				state, reward=episode[i]

				truncatedReturn+=gammaPower*reward
				gammaPower*=gamma

			for i in range(T-1):

				state, reward=episode[i]
				_state, _reward=episode[min(T-1, i+n)]

				_value=0

				if _state>=1 and _state<=self.n:

					_value=self.value[_state]

				group=(state-1)//(self.n//groups)

				weight[group]+=alpha*(truncatedReturn+gammaPower*_value-weight[group])

				truncatedReturn-=reward
				truncatedReturn+=_reward*gammaPower
				truncatedReturn/=gamma

			for state in range(1, self.n+1):

				self.value[state]=weight[(state-1)//(self.n//groups)]

			error+=np.sqrt(np.sum(np.power(self.value-trueValue, 2))/self.n)

		return error/10

def Left():

	global alpha

	states=1000
	episodes=100000
	runs=1

	ValueMonteCarlo=np.zeros(states+1)
	ValueMonteCarloFunctionApproximation=np.zeros(states+1)

	monteCarloTrueValue=RandomWalk(states)
	monteCarloTrueValue.dynamicProgrammingTrueValue()
	ValueMonteCarlo+=monteCarloTrueValue.value

	alpha=2e-4
	monteCarloFunctionApproximation=RandomWalk(states)
	monteCarloFunctionApproximation.TDZeroFunctionApproximation(episodes)
	ValueMonteCarloFunctionApproximation+=monteCarloFunctionApproximation.value


	plt.plot(ValueMonteCarlo, label='True Values')
	plt.plot(ValueMonteCarloFunctionApproximation, label='Values Using State Aggregation')

	plt.xlabel('State')
	plt.ylabel('Value')
	plt.title('TD(0) Function Approximation using State Aggregation')

	plt.xlim(1,1000)
	plt.ylim(-1,1)
	plt.legend()

	plt.savefig('example_9_2_Left.png')
	plt.close()

def Right():

	states=1000
	runs=100
	ns=np.power(2, np.arange(0, 10))
	alphas=np.arange(0.001, 1.001, 0.001)

	dummy=RandomWalk(states)
	episodes=[[dummy.generateEpisode() for i in range(10)] for i in range(runs)]
	dummy.dynamicProgrammingTrueValue(1e-2, 10)
	trueValue=dummy.value

	finalErrors=np.zeros((10,len(alphas)))

	for a in tqdm(range(len(alphas))):

		alpha=alphas[a]
		errors=np.zeros(10)

		for run in range(runs):

			for j,n in enumerate(ns):

				NstepTD=RandomWalk(states)
				errors[j]+=NstepTD.NStepTDFunctionApproximation(episodes[run], trueValue, 20, n, alpha)

		errors/=runs

		for j,n in enumerate(ns):

			finalErrors[j, a]=errors[j]

	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	fig.suptitle('N-Step TD using State Aggregation')
	plots=[]
	legends=[]

	for i,n in enumerate(ns):

		plot=ax.plot(finalErrors[i])[0]
		legends.append("N={}".format(ns[i]))
		plots.append(plot)

	ax.set_ylim(0.25,0.6)
	ax.set_xticks(np.arange(0,len(alphas)+len(alphas)/10,len(alphas)/10))
	ax.set_xticklabels(np.round(np.arange(0,1.1,0.1),2))
	ax.legend(plots, labels=legends, loc="center right", borderaxespad=-7.5)
	plt.subplots_adjust(right=0.85)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('RMS Error (averaged over 100 runs)')
	plt.savefig('example_9_2_Right.png')
	plt.close()

Left()
Right()