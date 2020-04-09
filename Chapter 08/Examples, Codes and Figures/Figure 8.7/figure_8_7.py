# I have taken a example in which there is a start state with only one action
# which leads to b next states with equal probability and random reward with mean 1.
# True value of all next states is 0 as they lead to termination with 0 reward.
# value of gamma=1

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Environment:

	def __init__(self, _b):

		self.b=_b
		self.rewards=np.random.normal(1, 1, self.b)
		val=np.sum(self.rewards)-self.rewards[-1]
		self.rewards[-1]=self.b-val

		if self.b==2:

			self.rewards[0]=0.7
			self.rewards[0]=1.3

	def getProbability(self, state, nxt, reward):

		if state==0 and nxt>0 and nxt<self.b+1 and reward==self.rewards[nxt-1]:

			return 1/self.b

		if state>0 and state<self.b+1 and nxt==-1 and reward==0:

			return 1

		return 0

	def getStartState(self):

		return 0

	def getNext(self, state):

		if state==0:

			nxt=np.random.randint(1,self.b+1)

			return nxt, self.rewards[nxt-1]

		else:

			return -1, 0

def expectedUpdate(environment):

	totalStates=environment.b+1
	Qvalue=np.zeros(totalStates)
	errors=[]

	for Updates in range(2):
		
		state=environment.getStartState()
		value=0

		for i in range(environment.b-1):
	
			errors.append(np.abs(1-Qvalue[state]))

		for nxt in range(totalStates):

			for reward in environment.rewards:

				p=environment.getProbability(state, nxt, reward)

				if p>0:

					value+=p*(reward+Qvalue[nxt])

		Qvalue[state]=value

		errors.append(np.abs(1-Qvalue[state]))

	return errors

def sampleUpdates(environment, episodes):
	
	Qvalue=0
	count=0
	errors=[]

	for episode in range(episodes):

		state=environment.getStartState()

		nxt, reward=environment.getNext(state)

		p=environment.getProbability(state, nxt, reward)

		if p>0:

			count+=1
			Qvalue+=(reward+0-Qvalue)/count
			errors.append(np.abs(1-Qvalue))

	return np.asarray(errors)

def Main():

	env=Environment(20)
	expected_errors=expectedUpdate(env)

	runs=1000

	envs=[Environment(b) for b in [2, 10, 100, 1000, 10000]]
	sample_errors=[np.zeros(2*en.b) for en in envs]

	for run in tqdm(range(runs)):

		for index, en in enumerate(envs):

			sample_errors[index]+=sampleUpdates(en, en.b*2)

	for er in sample_errors:

			er/=runs

	X1=np.linspace(2/40, 2, 40)
	plt.plot(X1, expected_errors, label='Expectation Update')

	for index, en in enumerate(envs):

		X=np.linspace(2/len(sample_errors[index]), 2, len(sample_errors[index]))
		plt.plot(X, sample_errors[index], label='b={}'.format(en.b))

	plt.xlabel('Number of Max computations')
	plt.xticks([0, 1, 2], ['0', 'b', '2b'])
	plt.ylabel('RMS Error')
	plt.title('Expectation vs Sampling')
	plt.legend()

	plt.savefig('figure_8_7.png')
	plt.show()
	plt.close()

Main()