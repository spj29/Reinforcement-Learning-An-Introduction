import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

gamma=0.95
alpha=1
eps=0.1
max_steps=7000
switch=1000

actions=[(0, 1), (0, -1), (1, 0), (-1, 0)]

class Maze:

	def __init__(self, _n, _m):

		self.N=_n
		self.M=_m
		self.valid=np.ones((_n, _m), dtype='int')
		self.target=(0, _m-1)
		self.start=(_n-1, 3)

	def getNext(self, x, y, a):

		dx, dy=actions[a]

		if x+dx>=self.N or x+dx<0 or y+dy>=self.M or y+dy<0:

			return (x,y)

		if self.valid[x+dx, y+dy]:

			return (x+dx, y+dy)

		else:

			return (x,y)

	def reset(self):

		self.valid=np.ones((self.N, self.M))
		
		blocks=[(3, i) for i in range(8)]

		for i, j in blocks:
			
			self.valid[i, j]=0		

class Model:

	def __init__(self, _k=0):

		self.model={}
		self.time=0
		self.k=_k

	def add(self, x, y, a, nx, ny, r):

		self.time+=1

		if self.k>0:

			if (x,y,a) not in self.model.keys():

				for i in range(len(actions)):

					self.model[(x,y,i)]=(x,y,0,1)

		self.model[(x,y,a)]=(nx, ny, r, self.time)

	def get_random(self):

		index=np.random.randint(len(self.model.keys()))
		key=list(self.model.keys())[index]

		nx, ny, r, time=self.model[key]
		r+=self.k*np.sqrt(self.time-time)

		return key, nx, ny, r

class DynaQ:

	def __init__(self, _n, _k, _N, _M):

		self.n=_n
		self.model=Model(_k)
		self.rewards=np.zeros(max_steps)
		self.Qvalue=np.zeros((_N, _M, len(actions)),dtype='float')

	def getAction(self, x, y):

		if np.random.rand()<=eps:

			return np.random.randint(len(actions))

		else:

			return np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[x, y] == np.max(self.Qvalue[x, y]))))

	def getActionExploratory(self, x, y):

		vals=np.zeros(len(actions), dtype='float')

		for a in range(len(actions)):

			if (x,y,a) in self.model.model.keys():

				nx, ny, r, time=self.model.model[(x,y,a)]
				vals[a]=self.Qvalue[x, y, a]+0.0001*np.sqrt(self.model.time-time)

			else:

				vals[a]=self.Qvalue[x, y, a]+0.0001*np.sqrt(self.model.time-0)
		
		return np.random.choice(np.ndarray.flatten(np.argwhere(vals == np.max(vals))))

	def performPlanning(self):

		for i in range(self.n):

			key, nx, ny, r=self.model.get_random()
			x, y, a=key

			self.Qvalue[x, y, a]+=alpha*(r+gamma*np.amax(self.Qvalue[nx, ny])-self.Qvalue[x, y, a])

	def work(self, maze, action_selection='e-greedy'):

		current=1
		maze.reset()

		while current<max_steps:

			currentX, currentY=maze.start

			while (currentX, currentY) != maze.target:

				if current == switch:

					maze.valid[maze.N-3, 0]=1
					maze.valid[maze.N-3, maze.M-1]=0

				currentA=0

				if action_selection=='e-greedy':
					
					currentA=self.getAction(currentX, currentY)
				
				else:

					currentA=self.getActionExploratory(currentX, currentY)
				
				newX, newY=maze.getNext(currentX, currentY, currentA)
				
				reward=0
				if (newX, newY) == maze.target:
					reward=1

				if current<max_steps:

					self.rewards[current]=self.rewards[current-1]+reward
				
				current+=1

				self.model.add(currentX, currentY, currentA, newX, newY, reward)

				self.Qvalue[currentX, currentY, currentA]+=alpha*(reward+gamma*np.amax(self.Qvalue[newX, newY])-self.Qvalue[currentX, currentY, currentA])

				(currentX, currentY)=(newX, newY)

				if self.n>0:
					
					self.performPlanning()

		return self.rewards

def Main():

	maze=Maze(6, 9)
	blocks=[(3, i) for i in range(8)]

	for i, j in blocks:
		
		maze.valid[i, j]=0

	runs=30

	rewardsDynaQ=np.zeros(max_steps)
	rewardsDynaQplus=np.zeros(max_steps)
	rewardsDynaQExploratory=np.zeros(max_steps)

	for run in tqdm(range(runs)):

		dynaQ=DynaQ(5, 0, maze.N, maze.M)
		dynaQplus=DynaQ(5, 0.0001, maze.N, maze.M)
		dynaQExploratory=DynaQ(5, 0, maze.N, maze.M)

		rewardsDynaQ+=dynaQ.work(maze)
		rewardsDynaQplus+=dynaQplus.work(maze)
		rewardsDynaQExploratory+=dynaQExploratory.work(maze, 'exploratory')

	rewardsDynaQ/=runs
	rewardsDynaQplus/=runs
	rewardsDynaQExploratory/=runs

	plt.plot(rewardsDynaQ, label='Dyna-Q')
	plt.plot(rewardsDynaQplus, label='Dyna-Q+')
	plt.plot(rewardsDynaQExploratory, label='Dyna-Q-Explore')

	plt.xlabel('Time Steps')
	plt.ylabel('Cumulative Reward')
	plt.title('Dyna-Q vs Dyna-Q+ vs Dyna-Q-Explore on Blocking Maze')
	plt.legend()

	plt.savefig('exercise_8_4.png')

Main()