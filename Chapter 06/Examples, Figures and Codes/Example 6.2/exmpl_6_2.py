import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_episode():

	episode=[]

	current=2

	while(current>=0 and current<5):

		episode.append(current)
		action=np.random.choice([-1,1])
		current+=action

	reward=0
	if current==5:
		reward=1

	return episode, reward

def MonteCarlo(episodes, alpha):

	true_value=np.arange(1, 6)/6
	value=np.ones(5)*0.5
	errors=np.zeros(episodes)

	for i in range(episodes):

		episode, reward=get_episode()

		for state in episode:

			value[state]+=alpha*(reward-value[state])

		errors[i]=np.sqrt(np.sum(np.power(true_value-value, 2))/5)

	return value, errors

def TDZero(episodes, alpha):

	true_value=np.arange(1, 6)/6
	value=np.ones(5)*0.5
	errors=np.zeros(episodes)

	for i in range(episodes):

		current=2
		while(current>=0 and current<5):

			action=np.random.choice([-1,1])

			nxt=current+action
			
			reward=0
			if nxt==5:
				reward=1

			value_next=0
			if nxt>=0 and nxt<5:
				value_next=value[nxt]

			value[current]+=alpha*(reward+value_next-value[current])

			current=nxt

		errors[i]=np.sqrt(np.sum(np.power(true_value-value, 2))/5)

	return value, errors

def PlotStateValue():
	
	true_value=np.arange(1, 6)/6

	for episodes in [0, 1, 10, 100]:

		value, errors=TDZero(episodes, 0.1)
		plt.plot(value, marker='o', label='Episodes: '+str(episodes))

	plt.plot(true_value, marker='o', label='True Values')

	plt.title('Values Estimated by TD(0)')
	plt.xlabel('State')
	plt.ylabel('Value')
	plt.xticks(np.arange(0,5), [chr(int(i)) for i in (np.ones(5)*ord('A')+np.arange(0,5))])
	plt.legend()
	plt.savefig('value.png')
	plt.close()

def PlotRMSError():

	true_value=np.arange(1, 6)/6

	runs=100
	episodes=100

	TDalphas=[0.05, 0.1, 0.15]
	MCalphas=[0.01, 0.02, 0.03, 0.04]

	# TD(0)
	for alpha in TDalphas:

		final_errors=np.zeros(episodes)
		
		for run in tqdm(range(runs)):

			value, errors=TDZero(episodes, alpha)

			final_errors+=errors

		final_errors/=runs

		plt.plot(final_errors, label='TD(0), Alpha= '+str(alpha))

	# Monte Carlo
	for alpha in MCalphas:

		final_errors=np.zeros(episodes)
		
		for run in tqdm(range(runs)):

			value, errors=MonteCarlo(episodes, alpha)

			final_errors+=errors

		final_errors/=runs

		plt.plot(final_errors, label='MC, Alpha= '+str(alpha))

	plt.legend()
	plt.xlabel('Episodes')
	plt.ylabel('RMS Error, averaged over states')
	plt.title('RMS Error, TD vs MC')

	plt.savefig('RMSError.png')
	plt.close()

def Main():

	PlotStateValue()
	PlotRMSError()

Main()