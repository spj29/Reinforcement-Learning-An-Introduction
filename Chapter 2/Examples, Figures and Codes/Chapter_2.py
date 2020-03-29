from numpy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

class Bandit:

	q=[0]
	Q=[0]
	H=[0]
	pr=[0]
	ucbQ=[0]
	N=[0]
	k=1
	c=2
	time=0
	reward=[]
	optimal=[]
	eps=0.5
	stationary=False
	algorithm=""
	alpha=0.1;
	totalReward=0
	keep_baseline=True

	def __init__(self ,_algorithm,_k,_stationary,_e,_a,):

		self.k=_k
		self.eps=_e
		if not _stationary:
			self.q=zeros(_k)
		else:
			self.q=random.normal(0,1,_k)
		self.Q=zeros(_k)
		self.ucbQ=[1000000000]*_k
		self.H=zeros(_k)
		self.pr=[1/_k]*_k
		self.N=zeros(_k,dtype=int)
		self.optimal=[]
		self.reward=[]
		self.algorithm=_algorithm
		self.alpha=_a
		self.time=0
		self.totalReward=0

	def ChooseGreedy(self):

		Mx=amax(self.Q)
		maxIndices=ndarray.flatten(argwhere(self.Q==Mx))
		return random.choice(maxIndices,1)[0]

	def ChooseRandom(self):

		return random.randint(0,self.k-1)

	def ChooseUCB(self):

		Mx=amax(self.ucbQ)
		maxIndices=ndarray.flatten(argwhere(self.ucbQ==Mx))
		return random.choice(maxIndices,1)[0]

	def ChooseGradient(self):
		return random.choice(self.k,1,p=self.pr)[0]

	def SampleAverage_PerformAction(self, _q = [],_ucb=False):

		self.q=add(self.q,_q)
		Next = 0

		if _ucb==True:
			Next=self.ChooseUCB()
		else:
			if random.randint(1,100) <= (self.eps*100):
				Next=self.ChooseRandom()
			else:
				Next=self.ChooseGreedy()

		if self.q[Next]==amax(self.q):
			self.optimal.append(100)
		else:
			self.optimal.append(0)

		Current_Reward=random.normal(self.q[Next],1)
		self.reward.append(Current_Reward)
		self.N[Next]+=1
		self.Q[Next]=self.Q[Next]+(Current_Reward-self.Q[Next])/self.N[Next]

	def ConstantStepSize_PerformAction(self, _q = [],_ucb=False):

		self.q=add(self.q,_q)
		Next = 0
		
		if _ucb==True:
			Next=self.ChooseUCB()
		else:
			if random.randint(1,100) <= (self.eps*100):
				Next=self.ChooseRandom()
			else:
				Next=self.ChooseGreedy()

		if self.q[Next]==amax(self.q):
			self.optimal.append(100)
		else:
			self.optimal.append(0)

		Current_Reward=random.normal(self.q[Next],1)
		self.reward.append(Current_Reward)
		self.N[Next]+=1
		self.Q[Next]=self.Q[Next]+self.alpha*(Current_Reward-self.Q[Next])

	def GradientAscent_PerformAction(self, _q = []):

		self.q=add(self.q,_q)
		Next = self.ChooseGradient()

		if self.q[Next]==amax(self.q):
			self.optimal.append(100)
		else:
			self.optimal.append(0)

		Current_Reward=random.normal(self.q[Next],1)
		self.reward.append(Current_Reward)
		self.N[Next]+=1

		Base=0
		if self.keep_baseline:
			Base=self.totalReward

		self.totalReward=self.totalReward+(Current_Reward-self.totalReward)/self.time

		for i in range(0,self.k):
			self.H[i]-=self.alpha*(Current_Reward-Base)*self.pr[i]
		self.H[Next]+=self.alpha*(Current_Reward-Base)

		eSum=sum(exp(self.H))
		self.pr=true_divide(exp(self.H),eSum);

	def Play(self,_q = [],_ucb=False):

		self.time+=1

		if self.algorithm=="SampleAverage":
			self.SampleAverage_PerformAction(_q,_ucb)
		elif self.algorithm=="ConstantStepSize":
			self.ConstantStepSize_PerformAction(_q,_ucb)
		elif self.algorithm=="GradientAscent":
			self.GradientAscent_PerformAction(_q)
		
		if _ucb:
			for i in range(0,self.k):
				if self.N[i]>0:
					self.ucbQ[i]=self.Q[i]+self.c*sqrt(log(self.time)/self.N[i])
				else:	
					self.ucbQ[i]=1000000000

def Main():

	eps=[1/128,1/64,1/32,1/16,1/8,1/4]
	alpha=[1/32,1/16,1/8,1/4,1/2,1,2,3]
	initQ=[1/4,1/2,1,2,4]
	c=[1/16,1/4,1/2,1,2,4]
	steps=1000
	agents=2000
	arms=10
	rewardsGreedy=[0]*len(eps)
	rewardsGreedyConstantStep=[0]*len(eps)
	rewardsGradient=[0]*len(alpha)
	rewardsOptimistic=[0]*len(initQ)
	rewardsUCB=[0]*len(c)

	for iterations in tqdm(range(0,agents)):

		banditsGreedy=[Bandit("SampleAverage",arms,True,e,0) for e in eps]
		banditsGreedyConstantStep=[Bandit("ConstantStepSize",arms,True,0,0.1) for e in eps]
		banditsGradient=[Bandit("GradientAscent",arms,True,0,a) for a in alpha]
		banditsOptimistic=[]
		for iq in initQ:
			temp=Bandit("ConstantStepSize",arms,True,0,0.1)
			temp.Q=[iq]*arms
			banditsOptimistic.append(temp)
		banditsUCB=[]
		for _c in c:
			temp=Bandit("SampleAverage",arms,True,0,0)
			temp.c=_c
			banditsUCB.append(temp)

		for j in range(0,steps):

			dq=zeros(arms);

			for i in range(0,len(banditsGreedy)):
				banditsGreedy[i].Play(dq);
			for i in range(0,len(banditsGreedyConstantStep)):
				banditsGreedyConstantStep[i].Play(dq);
			for i in range(0,len(banditsGradient)):
				banditsGradient[i].Play(dq);
			for i in range(0,len(banditsOptimistic)):
				banditsOptimistic[i].Play(dq);
			for i in range(0,len(banditsUCB)):
				banditsUCB[i].Play(dq,True);
		
		for i in range(0,len(banditsGreedy)):
			rewardsGreedy[i]+=mean(banditsGreedy[i].reward)
		for i in range(0,len(banditsGreedyConstantStep)):
			rewardsGreedyConstantStep[i]+=mean(banditsGreedyConstantStep[i].reward)
		for i in range(0,len(banditsGradient)):
			rewardsGradient[i]+=mean(banditsGradient[i].reward)
		for i in range(0,len(banditsOptimistic)):
			rewardsOptimistic[i]+=mean(banditsOptimistic[i].reward)
		for i in range(0,len(banditsUCB)):
			rewardsUCB[i]+=mean(banditsUCB[i].reward)

	for i in range(0,len(banditsGreedy)):
		rewardsGreedy[i]/=agents;
	for i in range(0,len(banditsGreedyConstantStep)):
		rewardsGreedyConstantStep[i]/=agents;
	for i in range(0,len(banditsGradient)):
		rewardsGradient[i]/=agents;
	for i in range(0,len(banditsOptimistic)):
		rewardsOptimistic[i]/=agents;
	for i in range(0,len(banditsUCB)):
		rewardsUCB[i]/=agents;

	plt.xscale("log",basex=2)
	plt.plot(eps,rewardsGreedy,label="Eps Greedy")
	plt.plot(eps,rewardsGreedyConstantStep,label="Eps Greedy (Constant Step Size)")
	plt.plot(alpha,rewardsGradient,label="Gradient Bandit")
	plt.plot(initQ,rewardsOptimistic,label="Optimistic Greedy")
	plt.plot(c,rewardsUCB,label="UCB")
	
	plt.xlabel("eps, alpha, C, Q_initial")
	plt.ylabel("Average Reward")
	plt.legend()

	plt.savefig('Exercise_2_1_1.png')

Main()