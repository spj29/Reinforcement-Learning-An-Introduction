import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

features=50
x_max=10

def square_wave(x):

	if x>x_max/4 and x<x_max*3/4:

		return 1

	return 0

def getExample():

	return np.random.uniform(0, x_max)

def getFeatureVector(s, featureWidth):

	x=np.zeros(features, dtype='float')
	d=(x_max-featureWidth)/(features-1)
	n=0

	for i in range(features):

		if s>=i*d and s<i*d+featureWidth:

			x[i]=1
			n+=1

	return x, n

def getEstimation(w, featureWidth):

	y=np.zeros(x_max*2000)

	for i in range(x_max*2000):

		x, n=getFeatureVector(i/2000, featureWidth)
		y[i]=np.dot(w, x)

	return y

def functionApproximation(featureWidth, examples, sample=[]):

	w=np.zeros(features, dtype='float')
	result=[]

	for example in tqdm(range(examples)):

		s=getExample()
		u=square_wave(s)
		x, n=getFeatureVector(s, featureWidth)
		alpha=0.2/n

		w=np.add(w, alpha*(u-np.dot(w, x))*x)

		if example+1 in sample:

			result.append(getEstimation(w, featureWidth))

	return result

def Main():

	small=x_max/20
	medium=x_max/5
	large=x_max*4/10
	examples=10240
	sample=[10, 40, 160, 640, 2560, 10240]

	result_small=functionApproximation(small, examples, sample)
	result_medium=functionApproximation(medium, examples, sample)
	result_large=functionApproximation(large, examples, sample)

	plt.figure(figsize=(15,25))
	index=1

	for i, ex in enumerate(sample):

		x=np.linspace(0, x_max, len(result_small[i]))
		plt.subplot(6, 3, index)
		plt.plot(x, result_small[i])
		plt.title('{} Examples. Width={}'.format(ex, small))
		index+=1

		x=np.linspace(0, x_max, len(result_medium[i]))
		plt.subplot(6, 3, index)
		plt.plot(x, result_medium[i])
		plt.title('{} Examples. Width={}'.format(ex, medium))
		index+=1

		x=np.linspace(0, x_max, len(result_large[i]))
		plt.subplot(6, 3, index)
		plt.plot(x, result_large[i])
		plt.title('{} Examples. Width={}'.format(ex, large))
		index+=1

	plt.savefig('figure_9_8.png')

Main()