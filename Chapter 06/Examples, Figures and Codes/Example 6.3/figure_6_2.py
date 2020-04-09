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

def MonteCarloBatch(episodes, alpha):

	true_value=np.arange(1, 6)/6
	value=np.ones(5)*0.5
	errors=np.zeros(episodes)
	history=[]

	for i in range(episodes):

		episode, reward=get_episode()

		for state in episode:

			history.append((state, reward))

		while 1:

			delta=np.zeros(5)
			
			for state, Return in history:

				delta[state]+=alpha*(Return-value[state])

			if np.sum(np.abs(delta))<1e-3:
				break

			value+=delta

		errors[i]=np.sqrt(np.sum(np.power(true_value-value, 2))/5)

	return value, errors

def TDZeroBatch(episodes, alpha):

	true_value=np.arange(1, 6)/6
	value=np.ones(5)*0.5
	errors=np.zeros(episodes)
	history=[]

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

			history.append((current, reward+value_next))

			current=nxt

		while 1:

			delta=np.zeros(5)
			
			for state, Return in history:

				delta[state]+=alpha*(Return-value[state])

			if np.sum(np.abs(delta))<1e-3:
				break

			value+=delta

		errors[i]=np.sqrt(np.sum(np.power(true_value-value, 2))/5)

	return value, errors

def PlotRMSError():

	runs=100
	episodes=100
	alpha=0.001

	TDRMSError=np.zeros(episodes)
	MCRMSError=np.zeros(episodes)

	# TD(0)
	for run in tqdm(range(runs)):

		value, errors=TDZeroBatch(episodes, alpha)
		TDRMSError+=errors

	TDRMSError/=runs

	# MC(0)
	for run in tqdm(range(runs)):

		value, errors=MonteCarloBatch(episodes, alpha)
		MCRMSError+=errors

	MCRMSError/=runs

	plt.plot(TDRMSError, label='TD(0)')
	plt.plot(MCRMSError, label='MC')

	plt.legend()
	plt.title('TD vs MC on Batch Processing')
	plt.xlabel('Episodes')
	plt.ylabel('RMS Error Averaged Over States')

	plt.savefig('RMS_batch.png')
	plt.close()

def Main():

	PlotRMSError()

Main()