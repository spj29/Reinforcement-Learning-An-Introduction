import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha=0.005
beta=0.05
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

def TDC(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	v=np.zeros(8)
	result=np.zeros((steps, 8), dtype='float')
	BE=np.zeros(steps, dtype='float')
	VE=np.zeros(steps, dtype='float')
	X=np.zeros((7,8))
	D=np.zeros((7,7))
	error=np.zeros(7)
	error_ve=np.zeros(7)

	for i in range(7):
		
		X[i]=getFeatureVector(i)
		D[i, i]=1/7

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
		delta=gamma*val_nxt-np.dot(w, feature)
		w=np.add(w, alpha*getImportanceRatio(action)*(delta*feature-gamma*getFeatureVector(nxt)*np.dot(v, feature)))
		v=np.add(v, beta*getImportanceRatio(action)*(delta-np.dot(v, feature))*feature)

		current=nxt
		result[step]+=w

		for state in range(7):

			error[state]=gamma*np.dot(w, getFeatureVector(6))-np.dot(w, getFeatureVector(state))
			error_ve[state]=np.power(np.dot(w, getFeatureVector(state)), 2)

		xtd=np.matmul(np.transpose(X),D)
		BE[step]=np.sqrt(np.dot(error, np.matmul(np.matmul(np.matmul(np.matmul(D,X),np.linalg.inv(np.matmul(xtd,X))),xtd),np.transpose(error))))	
		VE[step]=np.sqrt(np.sum(error_ve)/7)

	return result, BE, VE

def expectedTDC(steps):

	w=np.array([1.0,1.0,1.0,1.0,1.0,1.0,10.0,1.0])
	v=np.zeros(8)
	result=np.zeros((steps, 8), dtype='float')
	BE=np.zeros(steps, dtype='float')
	VE=np.zeros(steps, dtype='float')
	X=np.zeros((7,8))
	D=np.zeros((7,7))
	error=np.zeros(7)
	error_ve=np.zeros(7)

	for i in range(7):

		X[i]=getFeatureVector(i)
		D[i, i]=1/7

	for step in tqdm(range(steps)):

		temp=np.zeros(8)
		temp_v=np.zeros(8)
		feature=getFeatureVector(6)
		val=np.dot(w, feature)

		for state in range(7):

			feature=getFeatureVector(state)
			delta=gamma*val-np.dot(w, feature)
			temp=np.add(temp, delta*feature-gamma*getFeatureVector(6)*np.dot(v, feature))
			temp_v=np.add(temp_v, (delta-np.dot(v, feature))*feature)

		w=np.add(w, (alpha/7)*temp)
		v=np.add(v, (beta/7)*temp_v)
		result[step]+=w

		for state in range(7):

			error[state]=gamma*np.dot(w, getFeatureVector(6))-np.dot(w, getFeatureVector(state))
			error_ve[state]=np.power(np.dot(w, getFeatureVector(state)), 2)

		xtd=np.matmul(np.transpose(X),D)
		BE[step]=np.sqrt(np.dot(error, np.matmul(np.matmul(np.matmul(np.matmul(D,X),np.linalg.inv(np.matmul(xtd,X))),xtd),np.transpose(error))))	
		VE[step]=np.sqrt(np.sum(error_ve)/7)

	return result, BE, VE

def Left():

	steps=1000
	result, BE, VE=TDC(steps)

	plt.figure(figsize=(6,8))
	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.plot(BE, label='RMSBE')
	plt.plot(VE, label='RMSVE')

	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('TDC on Baird\'s Counterexample')
	plt.legend(loc="center right", borderaxespad=-7.5)
	plt.subplots_adjust(right=0.80)

	plt.savefig('left.png')
	plt.close()

def Right():

	steps=1000
	result, BE, VE=expectedTDC(steps)

	plt.figure(figsize=(6,8))
	for i in range(8):

		plt.plot(result[:,i], label=r'$w_{}$'.format(i+1))

	plt.plot(BE, label='RMSBE')
	plt.plot(VE, label='RMSVE')
	
	plt.xlabel('Steps')
	plt.ylabel('Value')
	plt.title('Expected TDC on Baird\'s Counterexample')
	plt.legend(loc="center right", borderaxespad=-7.5)
	plt.subplots_adjust(right=0.80)
	plt.savefig('right.png')
	plt.close()	

Left()
Right()