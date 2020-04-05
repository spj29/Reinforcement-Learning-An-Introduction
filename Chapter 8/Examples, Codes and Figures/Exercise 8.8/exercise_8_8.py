import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

max_steps=20000
sample=100
eps=0.1

class Environment:

	def __init__(self, _b, _states):

		self.states=_states
		self.b=_b
		self.nxt=np.random.randint(0, self.states, (self.states, 2, self.b))
		self.rewards=np.random.normal(0,1,(self.states, 2, self.b))

	def getNext(self, state, action):

		if np.random.rand()<0.1:

			return -1, 0

		index=np.random.randint(self.b)

		return self.nxt[state, action, index], self.rewards[state, action, index]

def evaluate(env, Qvalue):

	runs=1000
	answer=0

	for run in range(runs):

		state=0
		_return=0

		while state!=-1:

			action=np.argmax(Qvalue[state])

			state, reward=env.getNext(state, action)
			_return+=reward

		answer+=(_return-answer)/(run+1)

	return answer

def uniform(env):

	Qvalue=np.zeros((env.states,2))
	steps=0
	Value=[0]

	while steps<max_steps:

		state=np.random.randint(env.states)
		action=np.random.randint(2)

		if steps<max_steps:

			Qvalue[state, action]=0.9*(np.mean(env.rewards[state, action]+np.max(Qvalue[env.nxt[state, action],:], axis=1)))
			steps+=1

			if steps%sample==0:

				Value.append(evaluate(env, Qvalue))
				print('Uniform, Steps={}, Valu={}'.format(steps, Value[-1]))

	return np.asarray(Value)									

def on_policy(env):

	Qvalue=np.zeros((env.states,2))
	steps=0
	Value=[0]

	while steps<max_steps:

		state=0
		
		while state!=-1 and steps<max_steps:

			if np.random.rand()<eps:

				action=np.random.randint(2)

			else:

				action=np.argmax(Qvalue[state, :])

			Qvalue[state, action]=0.9*(np.mean(env.rewards[state, action]+np.max(Qvalue[env.nxt[state, action],:], axis=1)))
			
			steps+=1

			if steps%sample==0:

				Value.append(evaluate(env, Qvalue))
				print('On-Policy, Steps={}, Valu={}'.format(steps, Value[-1]))

			state, reward=env.getNext(state, action)
	
	return np.asarray(Value)						

def Main():

	global max_steps
	global sample

	max_steps=50000
	sample=100
	
	runs=20
	B=[3]

	values_uniform=[np.zeros((max_steps//sample)+1) for b in B]
	values_onpolicy=[np.zeros((max_steps//sample)+1) for b in B]
	
	for run in tqdm(range(runs)):

		envs=[Environment(b, 10000) for b in B]
		
		for i, en in enumerate(envs):

			values_uniform[i]+=uniform(en)
			values_onpolicy[i]+=on_policy(en)

	for i in range(len(B)):
		
		values_uniform[i]/=runs
		values_onpolicy[i]/=runs

	X=np.arange(0, max_steps+sample, sample)

	fig, ax = plt.subplots(1, 1, figsize=(13,6))
	fig.suptitle('Uniform vs Trajectory Sampling')
	plots=[]
	legends=[]

	for i in range(len(B)):

		plot=ax.plot(X, values_uniform[i])[0]
		legends.append('Uniform, b={}'.format(B[i]))
		plots.append(plot)

		plot=ax.plot(X, values_onpolicy[i])[0]
		legends.append('On-Policy, b={}'.format(B[i]))
		plots.append(plot)

	ax.legend(plots, labels=legends, loc="center right", borderaxespad=-13)
	plt.subplots_adjust(right=0.85)
	ax.set_xlabel('Number of Expected Updates')
	ax.set_ylabel('Value of Start State Under Greedy Policy')

	plt.savefig('exercise_8_8.png')
	plt.close()

Main()