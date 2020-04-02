import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

N=19
gamma=1

def generateEpisode():

	current=(N)//2
	episode=[]

	while(current>=0 and current<N):

		action=np.random.randint(2)
		_current=current

		if action==0:

			current-=1

		elif action==1:

			current+=1

		reward=0

		if current==N:

			reward=1

		elif current==-1:

			reward=-1

		episode.append((_current, action, reward))
	
	episode.append((current, 0, 0))

	return episode

def NStepTD(n, alpha, episodes):

	value=np.zeros(N)
	trueValue=np.arange(-18,20,2)/20
	errors=np.zeros(len(episodes))

	for index, episode in enumerate(episodes):

		T=len(episode)

		truncatedReturn=0
		gammaPower=1

		for i in range(T+n-1):

			t=i-n

			if t>=0:

				state, action, reward=episode[t]
				gammaPower=1
				TDerror=0

				for j in range(t,min(T-1,t+n)):

					_state, _action, _reward=episode[j]
				
					TDerror+=gammaPower*_reward
					gammaPower*=gamma

				_state, _action, _reward=episode[min(T-1, t+n)]

				_value=0

				if _state>=0 and _state<N:

					_value=value[_state]

				TDerror+=gammaPower*_value
				TDerror-=value[state]

				value[state]+=alpha*TDerror

		errors[index]=np.sqrt(np.sum(np.power(value-trueValue, 2))/N)

	return errors

def NstepTD_sum_of_TD_errors(n, alpha, episodes):

	value=np.zeros(N)
	trueValue=np.arange(-18,20,2)/20
	errors=np.zeros(len(episodes))

	for index, episode in enumerate(episodes):

		T=len(episode)
		delta=np.zeros(T)

		truncatedReturn=0
		gammaPower=1

		for i in range(T+n-1):

			t=i-n

			if t>=0:

				state, action, reward=episode[t]
				gammaPower=1
				TDerror=0

				for j in range(t,min(T-1,t+n)):

					_state, _action, _reward=episode[j]
				
					TDerror+=gammaPower*delta[j]
					gammaPower*=gamma

				value[state]+=alpha*TDerror

			if i<T-1:

				state, action, reward=episode[i]
				_state, _action, _reward=episode[min(T-1, i+1)]

				_value=0

				if _state>=0 and _state<N:

					_value=value[_state]

				delta[i]=reward+gamma*_value-value[state]

		errors[index]=np.sqrt(np.sum(np.power(value-trueValue, 2))/N)

	return errors

def Main():

	runs=100
	n=9
	alpha=0.3
	number_of_episodes=50
	episodes=[[generateEpisode() for i in range(number_of_episodes)] for i in range(runs)]

	error_NstepTD=np.zeros(number_of_episodes)
	error_NstepTD_constant_v=np.zeros(number_of_episodes)

	for run in tqdm(range(runs)):
		
		error_NstepTD+=NStepTD(n, alpha, episodes[run])
		error_NstepTD_constant_v+=NstepTD_sum_of_TD_errors(n, alpha, episodes[run])

	error_NstepTD/=runs
	error_NstepTD_constant_v/=runs

	plt.plot(error_NstepTD, label='{}-step TD with normal Error'.format(n))
	plt.plot(error_NstepTD_constant_v, label='{}-step TD with Error as sum of TD(0) errors'.format(n))

	plt.xlabel('Episode')
	plt.ylabel('RMS Error averaged over {} runs'.format(runs))
	plt.title(r'$\alpha={}, Episodes={}$'.format(alpha, number_of_episodes))
	plt.legend()

	plt.savefig('TESTExercise_7_2_{}_{}.png'.format(n, number_of_episodes))
	plt.close()

Main()