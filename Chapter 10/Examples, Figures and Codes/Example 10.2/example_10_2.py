import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

alpha=0.01
beta=0.01
eps=0.1
p_free=0.06
servers=10
priorities=[1, 2, 4, 8]

def accessControl(steps):

	Qvalue=np.zeros((servers+1, len(priorities), 2), dtype='float')
	r_average=0
	active=servers
	p=np.random.randint(len(priorities))
	a=0

	if active>0:

		if np.random.binomial(1, eps)>0:

			a=np.random.randint(2)

		else:

			a=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[active, p]==np.max(Qvalue[active, p]))))

	for step in tqdm(range(steps)):

		reward=a*priorities[p]
		free=np.random.binomial(servers-active, p_free)
		nxt_p=np.random.randint(len(priorities))
		nxt_active=active-a+free
		nxt_a=0
	
		if nxt_active>0:

			if np.random.binomial(1, eps)>0:

				nxt_a=np.random.randint(2)

			else:

				nxt_a=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[nxt_active, nxt_p]==np.max(Qvalue[nxt_active, nxt_p]))))

		delta=reward-r_average+Qvalue[nxt_active, nxt_p, nxt_a]-Qvalue[active, p, a]
		Qvalue[active, p, a]+=alpha*delta
		r_average+=beta*delta

		active, p, a=(nxt_active, nxt_p, nxt_a)

	print(r_average)
	return Qvalue

def Main():

	steps=3000000
	
	Qvalue=accessControl(steps)
	policy=np.argmax(Qvalue, axis=-1)

	plt.matshow(np.transpose(policy[1:servers+1,:]))
	plt.xlabel('Number of Free Servers')
	plt.xticks(np.arange(10), np.arange(1, 11))
	plt.ylabel('Priority')
	plt.yticks(np.arange(4), priorities)
	plt.savefig('Policy.png')
	plt.close()

	Qvalue[0,:,1]=-np.inf

	for i, p in enumerate(priorities):

		plt.plot(np.max(Qvalue[:, i, :], axis=-1), label='Priority={}'.format(p))

	plt.title('Action Values')
	plt.ylabel('Differential Action Value')
	plt.xlabel('Number of Free Servers')
	plt.legend()

	plt.savefig('qvalue.png')
	plt.close()

Main()
