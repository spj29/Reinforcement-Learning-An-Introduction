import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha=0.03
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

def expectedEmphaticTD(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	result=np.zeros((steps, 8), dtype='float')
	VE=np.zeros(steps, dtype='float')
	error_ve=np.zeros(7)
	M=0
	I=1

	for step in tqdm(range(steps)):

		temp=np.zeros(8)
		feature=getFeatureVector(6)
		val=np.dot(w, feature)

		for state in range(7):

			feature=getFeatureVector(state)
			delta=gamma*val-np.dot(w, feature)
			rho=7
			
			if state!=6:

				rho=0

			temp=np.add(temp, (gamma*rho*M+I)*delta*feature)

		w=np.add(w, (alpha/7)*temp)
		M=gamma*M+I
		result[step]+=w

		for state in range(7):

			error_ve[state]=np.power(np.dot(w, getFeatureVector(state)), 2)

		VE[step]=np.sqrt(np.sum(error_ve)/7)

	return result, VE

def Main():

	steps=1000
	result, VE=expectedEmphaticTD(steps)

	plt.figure(figsize=(6,5))
	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.plot(VE, label='RMSVE')
	
	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('Expected Emphatic-TD on Baird\'s Counterexample')
	plt.legend(loc="center right", borderaxespad=-7.5)
	plt.subplots_adjust(right=0.80)
	plt.savefig('figure_11_6.png')
	plt.close()	

Main()