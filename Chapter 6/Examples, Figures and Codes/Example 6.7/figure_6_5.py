import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

eps=0.1
alpha=0.1

def getAction(current, Qvalue):

	reward=0
	action=0

	if current==1:

		reward=np.random.normal(-0.1, 1)

	if np.random.rand()<=eps:

			action=np.random.choice([-1, 1])

	else:

		action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[current]==np.amax(Qvalue[current]))))

	return action, reward

def QLearning(episodes):

	Qvalue=[]
	Qvalue.append([0])
	Qvalue.append([0]*20)
	Qvalue.append([0]*2)
	Qvalue.append([0])

	left=np.zeros(episodes)

	for episode in range(episodes):

		current=2

		while(current>=1 and current<=2):

			action, reward=getAction(current, Qvalue)

			nxt=current

			if current==2 and action==0:
				
				left[episode]=1


			if current==1:

				nxt=0

			else:

				if action==0:

					nxt=1

				else:

					nxt=3


			Qvalue[current][action]+=alpha*(reward+np.amax(Qvalue[nxt])-Qvalue[current][action])

			current=nxt

	return left

def doubleQLearning(episodes):

	Qvalue_1=[]
	Qvalue_1.append([0])
	Qvalue_1.append([0]*20)
	Qvalue_1.append([0]*2)
	Qvalue_1.append([0])
	
	Qvalue_2=[]
	Qvalue_2.append([0])
	Qvalue_2.append([0]*20)
	Qvalue_2.append([0]*2)
	Qvalue_2.append([0])
	
	left=np.zeros(episodes)

	for episode in range(episodes):

		current=2

		while(current>=1 and current<=2):

			action, reward=getAction(current, Qvalue_1+Qvalue_2)

			nxt=current

			if current==2 and action==0:
				
				left[episode]=1


			if current==1:

				nxt=0

			else:

				if action==0:

					nxt=1

				else:

					nxt=3

			if np.random.randint(2)==0:
				
				Qvalue_1[current][action]+=alpha*(reward+Qvalue_1[nxt][np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue_2[nxt]==np.amax(Qvalue_2[nxt]))))]-Qvalue_1[current][action])

			else:

				Qvalue_2[current][action]+=alpha*(reward+Qvalue_2[nxt][np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue_1[nxt]==np.amax(Qvalue_1[nxt]))))]-Qvalue_2[current][action])
			
			current=nxt

	return left


def Main():

	runs=10000
	episodes=300

	QLeft=np.zeros(episodes)
	dQLeft=np.zeros(episodes)

	for run in tqdm(range(runs)):

		QLeft+=QLearning(episodes)
		dQLeft+=doubleQLearning(episodes)

	QLeft*=100
	QLeft/=runs

	dQLeft*=100
	dQLeft/=runs

	plt.plot(QLeft, label='Q Learning')
	plt.plot(dQLeft, label='Double Q Learning')

	plt.xlabel('Episodes')
	plt.ylabel('% Left action from A')

	plt.legend()

	plt.savefig('fig_6_5.png')

Main()