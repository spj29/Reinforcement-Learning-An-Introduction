import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha=0.01
gamma=0.99
b=np.zeros(2)
pi=np.zeros(2)
actions=[0, 1]
b[0]=6/7
b[1]=1/7
pi[1]=1

def getFeatureVector(state):

	x=np.zeros(8)

	if state==6:

		x[6]=1
		x[7]=2

	else:

		x[state]=2
		x[7]=1

	return x

def getImportanceRatio(a):

	return pi[a]/b[a]

def getNext(action):

	nxt=0

	if action==0:

		nxt=np.random.randint(6)

	else:

		nxt=6

	return nxt

def semiGradientQ(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	result=np.zeros((steps, 8), dtype='float')

	current=np.random.randint(7)

	for step in tqdm(range(steps)):

		action=np.random.choice(actions, p=b)
		nxt=getNext(action)

		vals=[np.dot(w, getFeatureVector(getNext(a))) for a in actions]

		feature=getFeatureVector(nxt)
		val_nxt=np.dot(w, feature)

		w=np.add(w, alpha*(gamma*np.max(vals)-val_nxt)*feature)

		current=nxt
		result[step]+=w

	return result

def Main():

	steps=1000
	result=semiGradientQ(steps)

	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('Semi Gradient Q-Learning on Baird\'s Counterexample')
	plt.legend()
	plt.savefig('exercise_11_3.png')
	plt.close()	

Main()