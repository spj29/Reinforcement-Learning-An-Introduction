import numpy as np
import matplotlib.pyplot as plt

target=100
target_reward=1
ph=0.55

class Gambler:

	def __init__(self):
		self.value=np.zeros((target))
		self.policy=np.zeros((target))
		self.value_history=[]

	def EvaluateState(self, i):

		max_stake=min(i,target-i)
		returns=np.zeros(max_stake+1)

		for stake in range(max_stake+1):

			returns[stake]=(1.0-ph)*self.value[i-stake]
			if i+stake==target:
				returns[stake]+=ph*target_reward
			else:
				returns[stake]+=ph*self.value[i+stake]

		return np.amax(returns),np.argmax(np.round(returns[1:], 5))+1

	def ValueIteration(self):

		while True:

			old_value=np.copy(self.value)
			self.value_history.append(old_value)

			for i in range(1,target):

				self.value[i], self.policy[i]=self.EvaluateState(i)


			delta=np.abs(old_value-self.value).max()

			print("Delta: "+str(delta))

			if(delta<1e-9):
				self.value_history.append(self.value)
				break

def Main():

	g=Gambler()
	g.ValueIteration()

	for i in range(1,min(15,len(g.value_history))):
		plt.plot(g.value_history[i],label="Sweep: "+str(i))
	
	plt.xlabel("Current Balance")
	plt.ylabel("Expected Return")
	plt.title('Probability Head: {}'.format(ph))

	plt.legend()
	plt.savefig('Value_{}.png'.format(ph))
	plt.close()

	plt.bar(np.arange(target),g.policy)
	plt.xlabel("Current Balance")
	plt.ylabel("Stake")
	plt.title('Probability Head: {}'.format(ph))

	plt.savefig('Policy_{}.png'.format(ph))
	plt.close()

Main()