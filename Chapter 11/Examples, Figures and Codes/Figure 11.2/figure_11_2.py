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

def semiGradientOffPolicyTD(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	result=np.zeros((steps, 8), dtype='float')

	current=np.random.randint(7)

	for step in tqdm(range(steps)):

		action=np.random.choice(actions, p=b)

		nxt=0

		if action==0:

			nxt=np.random.randint(6)

		else:

			nxt=6

		feature=getFeatureVector(nxt)
		val_nxt=np.dot(w, feature)

		feature=getFeatureVector(current)
		w=np.add(w, alpha*getImportanceRatio(action)*(gamma*val_nxt-np.dot(w, feature))*feature)

		current=nxt
		result[step]+=w

	return result

def semiGradientDP(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	result=np.zeros((steps, 8), dtype='float')

	for step in tqdm(range(steps)):

		temp=np.zeros(8)
		feature=getFeatureVector(6)
		val=np.dot(w, feature)

		for state in range(7):

			feature=getFeatureVector(state)
			temp=np.add(temp, (gamma*val-np.dot(w, feature))*feature)

		w=np.add(w, (alpha/7)*temp)
		result[step]+=w

	return result

def Left():

	steps=1000
	result=semiGradientOffPolicyTD(steps)

	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('Semi Gradient Off Policy TD on Baird\'s Counterexample')
	plt.legend()
	plt.savefig('left.png')
	plt.close()

def Right():

	steps=1000
	result=semiGradientDP(steps)

	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('Semi Gradient DP on Baird\'s Counterexample')
	plt.legend()
	plt.savefig('right.png')
	plt.close()	

Left()
Right()