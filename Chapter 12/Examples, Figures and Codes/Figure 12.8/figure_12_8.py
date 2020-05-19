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

def getFeatureVector(state):

	x=np.zeros(N)
	x[state]=1

	return x;

def trueOnlineTDLambda(lamda, alpha, episodes):

	value=np.zeros(N)
	trueValue=np.arange(-18,20,2)/20
	errors=0
	X=[getFeatureVector(i) for i in range(N)]

	for index, episode in enumerate(episodes):

		T=len(episode)

		truncatedReturn=0
		gammaPower=1
		z=np.zeros(N)
		Vold=0

		for i in range(T-1):

			state, action, reward=episode[i]

			_state, __, _=episode[i+1]

			_value=0

			if _state>=0 and _state<N:

				_value=np.dot(value, X[_state])

			val=np.dot(value, X[state])

			TDerror=reward+gamma*_value-val

			z=np.add(z*lamda*gamma, (1-alpha*gamma*lamda*np.dot(z,X[state]))*X[state])

			value=np.add(value, alpha*(TDerror+val-Vold)*z)
			value=np.add(value, -alpha*(val-Vold)*X[state])

			Vold=_value

		errors+=np.sqrt(np.sum(np.power(value-trueValue, 2))/N)

	return errors/10

def Main():

	runs=100
	ns=[0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
	alphas=np.arange(0.001, 1.001, 0.01)
	episodes=[[generateEpisode() for i in range(10)] for i in range(runs)]

	finalErrors=np.zeros((10,len(alphas)))

	for a in tqdm(range(len(alphas))):

		alpha=alphas[a]

		errors=np.zeros(10)

		for run in range(runs):

			for j,lamda in enumerate(ns):

				errors[j]+=trueOnlineTDLambda(lamda, alpha, episodes[run])

		errors/=runs

		for j,n in enumerate(ns):

			finalErrors[j, a]=errors[j]

	fig, ax = plt.subplots(1, 1, figsize=(10,6))
	fig.suptitle(r'$True \ Online \ TD(\lambda) \  Algorithm$')
	plots=[]
	legends=[]

	for i,n in enumerate(ns):

		plot=ax.plot(finalErrors[i])[0]
		legends.append(r'$\lambda={}$'.format(ns[i]))
		plots.append(plot)

	ax.set_ylim(0.25,0.6)
	ax.set_xticks(np.arange(0,len(alphas)+len(alphas)/10,len(alphas)/10))
	ax.set_xticklabels(np.round(np.arange(0,1.1,0.1),2))
	ax.legend(plots, labels=legends, loc="center right", borderaxespad=-9.5)
	plt.subplots_adjust(right=0.85)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('RMS Error (averaged over {} runs)'.format(runs))
	plt.savefig('figure_12_8.png')
	plt.close()

Main()