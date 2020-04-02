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

		else:

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
	errors=0

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

		errors+=np.sqrt(np.sum(np.power(value-trueValue, 2))/N)

	return errors/10

def Main():

	runs=100
	ns=np.power(2, np.arange(0, 10))
	alphas=np.arange(0.001, 1.001, 0.001)
	episodes=[[generateEpisode() for i in range(10)] for i in range(runs)]

	finalErrors=np.zeros((10,len(alphas)))

	for a in range(len(alphas)):

		alpha=alphas[a]
		
		print('Alpha={}'.format(alpha))

		errors=np.zeros(10)

		for run in tqdm(range(runs)):

			for j,n in enumerate(ns):

				errors[j]+=NStepTD(n, alpha, episodes[run])

		errors/=runs

		for j,n in enumerate(ns):

			finalErrors[j, a]=errors[j]

	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	fig.suptitle('Comparison of N-Step TD Algorithms')
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
	plt.savefig('TESTexmpl_7_1.png')
	plt.close()

Main()