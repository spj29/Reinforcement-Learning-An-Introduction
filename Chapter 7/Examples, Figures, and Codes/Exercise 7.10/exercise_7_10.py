import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N=19
gamma=1

probab=np.ones((N+1,2),dtype='float')

for i in range(N+1):
	
	probab[i, 0]=np.random.randint(100)/100
	probab[i, 1]=1-probab[i, 0]

probab[0, 0]=0
probab[0, 1]=1

def samplingRatio(state, action):

	return probab[state, action]*2

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

def NStepTD_Off_Policy(n, alpha, episodes, trueValue):

	value=np.zeros(N+1,dtype='float')
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
				importanceSampling=1

				for j in range(t,min(T-1,t+n)):

					_state, _action, _reward=episode[j]
				
					TDerror+=gammaPower*_reward
					importanceSampling*=samplingRatio(_state, _action)
					gammaPower*=gamma

				_state, _action, _reward=episode[min(T-1, t+n)]

				_value=value[_state]

				TDerror+=gammaPower*_value
				TDerror-=value[state]

				value[state]+=alpha*importanceSampling*TDerror

		errors[index]=np.sqrt(np.sum(np.power(value-trueValue, 2))/(N+1))

	return errors

def NStepTD_Off_Policy_Control_Variate(n, alpha, episodes, trueValue):

	value=np.zeros(N+1,dtype='float')
	errors=np.zeros(len(episodes))

	for index, episode in enumerate(episodes):

		T=len(episode)

		truncatedReturn=0
		gammaPower=1

		for i in range(T+n-1):

			t=i-n

			if t>=0:

				_state, _action, _reward=episode[min(T-1, t+n)]

				TDerror=0
				
				if _state>=0 and _state<N:
					
					TDerror=value[_state]

				state, action, reward=episode[t]

				for j in range(min(T-1,t+n)-1,t-1,-1):

					_state, _action, _reward=episode[j]
					rho=samplingRatio(_state, _action)
					TDerror=rho*(_reward+gamma*TDerror)+(1-rho)*value[_state]

				TDerror-=value[state]

				value[state]+=alpha*TDerror

		errors[index]=np.sqrt(np.sum(np.power(value-trueValue, 2))/(N+1))

	return errors

def trueValueDP():

	value=np.zeros(N+1,dtype='float')
	value[N]=0

	while 1:

		total_delta=0

		for state in range(N):

			dv=-value[state]

			for action in [0, 1]:

				nxt=state
				if action==0:
					nxt=state-1
				else:
					nxt=state+1
				
				reward=0
				_value=0
				if nxt==N:
					reward=1
				elif nxt==-1:
					reward=-1
				if nxt>=0 and nxt<=N:
					_value=value[nxt]

				dv+=probab[state, action]*(reward+_value)

			value[state]+=dv
			total_delta+=np.abs(dv)

		print(total_delta)

		if total_delta<1e-4:
			break

	return value

def Main():
	
	n=5
	alpha=0.1
	episodes=100
	runs=100
	episode=[[generateEpisode() for i in range(episodes)] for run in range(runs)]
	trueValues=trueValueDP()

	print(trueValues)
	
	errors=np.zeros(episodes)
	errors_cv=np.zeros(episodes)

	for run in tqdm(range(runs)):

		errors+=NStepTD_Off_Policy(n, alpha, episode[run], trueValues)
		errors_cv+=NStepTD_Off_Policy_Control_Variate(n, alpha, episode[run], trueValues)

	errors/=runs
	errors_cv/=runs
	plt.plot(errors, label='Off Policy without Control Variate')
	plt.plot(errors_cv, label='Off Policy with Control Variate')
	plt.xlabel('Episodes')
	plt.ylabel('RMS Error averaged over 100 runs')
	plt.legend()
	plt.savefig('exercise_7_10_{}_{}.png'.format(alpha, n))
	plt.close()

Main()


