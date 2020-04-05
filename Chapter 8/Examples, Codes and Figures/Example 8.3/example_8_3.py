import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

gamma=0.95
alpha=1
eps=0.1
max_steps=6000
switch=3000

actions=[(0, 1), (0, -1), (1, 0), (-1, 0)]

N=6
M=9

maze=np.ones((N, M),dtype='int')
newmaze=np.ones((N,M),dtype='int')
oldmaze=np.ones((N,M),dtype='int')

target=(0, M-1)
start=(N-1, 3)

def getNext(x, y, a):

	dx, dy=actions[a]

	if x+dx>=N or x+dx<0 or y+dy>=M or y+dy<0:

		return (x,y)

	if maze[x+dx, y+dy]==1:

		return (x+dx, y+dy)

	else:

		return (x,y)

class DynaQ:

	def __init__(self, _n, _k=0):

		self.n=_n
		self.k=_k
		self.rewards=np.zeros(max_steps)
		self.Qvalue=np.zeros((N, M, len(actions)),dtype='float')
		self.model={}
		self.time=0

	def getAction(self, x, y):

		if np.random.rand()<=eps:

			return np.random.randint(len(actions))

		else:

			return np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[x, y] == np.max(self.Qvalue[x, y]))))

	def performPlanning(self):

		choices=list(self.model.keys())
		indices=np.random.randint(0, len(choices), self.n)

		SA=[choices[i] for i in indices]

		for x, y, a in SA:

			r, nx, ny, time=self.model[(x, y, a)]
			r+=self.k*np.sqrt(self.time-time)

			self.Qvalue[x, y, a]+=alpha*(r+gamma*np.amax(self.Qvalue[nx, ny])-self.Qvalue[x, y, a])

	def work(self):

		global maze
		maze=np.copy(oldmaze)

		current=1

		while self.time<max_steps:

			currentX, currentY=start

			while (currentX, currentY) != target:

				if current == switch:

					maze=np.copy(newmaze)

				currentA=self.getAction(currentX, currentY)
				newX, newY=getNext(currentX, currentY, currentA)

				reward=0
				if (newX, newY) == target:
					reward=1

				if current<max_steps:

					self.rewards[current]=self.rewards[current-1]+reward
				
				self.time+=1
				current+=1

				if self.k>0:
	
					if (currentX, currentY, currentA) not in self.model.keys():

						for a in range(len(actions)):

							self.model[(currentX, currentY, a)]=(0, currentX, currentY, 1)

				self.model[(currentX, currentY, currentA)]=(reward, newX, newY, self.time)

				self.Qvalue[currentX, currentY, currentA]+=alpha*(reward+gamma*np.amax(self.Qvalue[newX, newY])-self.Qvalue[currentX, currentY, currentA])

				(currentX, currentY)=(newX, newY)

				if self.n>0:
					
					self.performPlanning()

		return self.rewards

def Main():

	for i in range(1,M):

		oldmaze[N-3, i]=0

	for i in range(1,M-1):

		newmaze[N-3, i]=0

	runs=30

	rewardsDynaQ=np.zeros(max_steps)
	rewardsDynaQplus=np.zeros(max_steps)

	for run in tqdm(range(runs)):

		dynaQ=DynaQ(50,0)
		dynaQplus=DynaQ(50,0.001)

		rewardsDynaQ+=dynaQ.work()
		rewardsDynaQplus+=dynaQplus.work()

	rewardsDynaQ/=runs
	rewardsDynaQplus/=runs

	plt.plot(rewardsDynaQ, label='Dyna-Q')
	plt.plot(rewardsDynaQplus, label='Dyna-Q+')

	plt.xlabel('Time Steps')
	plt.ylabel('Cumulative Reward')
	plt.title('Dyna-Q vs Dyna-Q+ on Changing Blocking Maze')
	plt.legend()

	plt.savefig('figure_8_5.png')

Main()