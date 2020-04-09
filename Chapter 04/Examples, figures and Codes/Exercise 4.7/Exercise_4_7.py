import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import multiprocessing as mp
from functools import partial
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
import itertools
import time
import math

max_cars_at_a=20
max_cars_at_b=20
expectedRental_a=3
expectedRental_b=4
expectedReturn_a=3
expectedReturn_b=2
max_rental=10
max_return=10
rental_reward=10
transport_cost=2
parkingcost=4
save_poisson=dict()

def getProbability(n,lam):

	key=10*n+lam
	if key not in save_poisson:
		save_poisson[key]=math.exp(-lam)*math.pow(lam, n)/math.factorial(n)

	return save_poisson[key]

class CarRental:

	def __init__(self, _max_cars_at_a, _max_cars_at_b, _discount):

		self.policy=np.zeros((_max_cars_at_a+1, _max_cars_at_b+1), dtype=np.int)
		self.value=np.zeros(self.policy.shape)
		self.discount=_discount

	def EvaluateState(self, i, j, action, value):
		
		reward=0
		answer=0

		if action>0:
			answer-=(action-1)*transport_cost
		else:
			answer-=abs(action)*transport_cost

		for rent_at_a in range(0,max_rental+1):
			for rent_at_b in range(0,max_rental+1):
				probabilityRental=getProbability(rent_at_a,expectedRental_a)*getProbability(rent_at_b,expectedRental_b)

				cars_at_a=min(max_cars_at_a,i-action);
				cars_at_b=min(max_cars_at_b,j+action);				

				acutal_rent_at_a=min(rent_at_a,cars_at_a)
				acutal_rent_at_b=min(rent_at_b,cars_at_b)

				reward=(acutal_rent_at_b+acutal_rent_at_a)*rental_reward

				if cars_at_a>=10:
					reward-=parkingcost
				if cars_at_b>=10:
					reward-=parkingcost

				cars_at_a-=acutal_rent_at_a
				cars_at_b-=acutal_rent_at_b

				for return_at_a in range(0,max_return+1):
					for return_at_b in range(0,max_return+1):

						probabilityAction=probabilityRental*getProbability(return_at_a,expectedReturn_a)*getProbability(return_at_b,expectedReturn_b)

						new_cars_at_a=min(cars_at_a+return_at_a,max_cars_at_a)
						new_cars_at_b=min(cars_at_b+return_at_b,max_cars_at_b)

						answer+=probabilityAction*(reward+self.discount*value[new_cars_at_a][new_cars_at_b])

		return answer

	def PolicyEvaluation(self, value, policy):

		global max_cars_at_a

		while True:
			time_1=time.time()
			old_value=np.copy(value)

			results=[]
			k = np.arange(max_cars_at_a + 1)
			states = ((i, j) for i, j in itertools.product(k, k))

			with mp.Pool(processes=8) as p:

				func=partial(self.EvaluateState_PE, value, policy)
				results=p.map(func, states)

			for v,i,j in results:
				value[i][j]=v

			delta=np.abs(old_value-value).max()
			print("Delta: "+str(delta))
			time_2=time.time()
			print("time: "+str(time_2-time_1))

			if delta<1e-2:
				break

		return value

	def PolicyImprovement(self, value, policy, actions):

		old_policy=np.copy(policy)
		isStable=True

		returns=np.zeros((max_cars_at_a+1,max_cars_at_b+1, np.size(actions)))

		for a in actions:

			with mp.Pool(processes=8) as p:

				k=np.arange(0,max_cars_at_a+1)
				states=((i,j) for i, j in itertools.product(k,k))

				func=partial(self.EvaluateState_PI, value, a)
				results=p.map(func, states)

				for v,i,j in results:
					returns[i][j][a+5]=v

		for i in range(0,max_cars_at_a+1):
			for j in range(0,max_cars_at_b+1):
				policy[i][j]=actions[np.argmax(returns[i][j])]

		if (old_policy!=policy).sum()>0:
			isStable=False
					
		return isStable, policy

	def EvaluateState_PE(self, value, policy, state):

		action=policy[state[0],state[1]]
		answer=self.EvaluateState(state[0],state[1],action,value)

		return answer,state[0],state[1]

	def EvaluateState_PI(self, value, action, state):

		answer=-np.inf
		if state[0]-action>=0 and state[1]+action>=0:
			answer=self.EvaluateState(state[0],state[1],action,value)

		return answer,state[0],state[1]

	def PlotValue(self):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		X=np.arange(0,max_cars_at_a+1)
		Y=np.arange(0,max_cars_at_a+1)
		X,Y=np.meshgrid(X,Y)
		ax.plot_wireframe(X, Y, np.transpose(self.value), rstride=1, cstride=1)
		ax.set_xlabel("#Cars at second location")
		ax.set_ylabel("#Cars at first location")
		ax.set_zlabel("State Value")
		ax.set_xticks([i for i in range(0,25,5)])
		ax.set_yticks([i for i in range(0,25,5)])
		plt.savefig('StateValueEx.png')
		plt.close()

	def PlotPolicy(self, iterations):

		fig1 = sns.heatmap(np.flipud(self.policy), cmap="YlGnBu")
		fig1.set_ylabel('# cars at first location')
		fig1.set_yticks(list(reversed(range(max_cars_at_a + 1))))
		fig1.set_xlabel('# cars at second location')
		fig1.set_title('policy {}'.format(iterations))
		plt.savefig('Policy_Ex_{}.png'.format(iterations))
		plt.close()

	def PolicyIteration(self):

		actions=[i for i in range(-5,6)]

		begin=time.time()

		iterations=0
		self.PlotPolicy(iterations)

		while True:

			self.value=self.PolicyEvaluation(self.value,self.policy)
			isStable, self.policy=self.PolicyImprovement(self.value, self.policy, actions)

			iterations+=1
			self.PlotPolicy(iterations)

			print("Policy Done")

			if isStable:
				self.PlotValue()
				break

		elapsed=time.time()-begin
		print("Done in : "+str(elapsed))

if __name__ == '__main__':

	cr=CarRental(20,20,0.9)
	cr.PolicyIteration()

	print(cr.policy)