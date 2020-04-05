import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from queue import Queue
import heapq

gamma=0.95
episodes=50
alpha=0.5
eps=0.1

actions=[(0, 1), (0, -1), (1, 0), (-1, 0)]

class Maze:

	def __init__(self):

		self.N=6
		self.M=9
		self.resolution=1

		self.maze=np.ones((self.N, self.M), dtype='int')
		self.maze[0, 7]=0
		self.maze[1, 7]=0
		self.maze[2, 7]=0
		self.maze[1, 2]=0
		self.maze[2, 2]=0
		self.maze[3, 2]=0
		self.maze[4, 5]=0

		self.target=(0, 8)
		self.start=(2, 0)

	def getNext(self, x, y, a):

		dx, dy=actions[a]

		if x+dx>=self.N or x+dx<0 or y+dy>=self.M or y+dy<0:

			return (x,y)

		if self.maze[x+dx, y+dy]:

			return (x+dx, y+dy)

		else:

			return (x,y)

	def Magnify(self, factor=1):

		self.resolution*=factor
		new_obstacles=[]

		for i in range(self.N):

			for j in range(self.M):

				if self.maze[i, j]==0:

					for di in range(0, factor):
						
						for dj in range(0, factor):

							new_obstacles.append((i*factor+di, j*factor+dj))

		si, sj=self.start
		ei, ej=self.target
		self.start=(si*factor, sj*factor)
		self.target=(ei*factor, ej*factor)
		self.N*=factor
		self.M*=factor

		self.maze=np.ones((self.N, self.M), dtype='int')

		for i, j in new_obstacles:

			self.maze[i, j]=0

	def getMinSteps(self):

		answer=0
		dist=np.ones((self.N,self.M), dtype='int')*1000000000

		si, sj=self.start
		ei, ej=self.target

		dist[si, sj]=0

		q=Queue()

		q.put(self.start)

		while not q.empty():

			current=q.get()
			i, j=current

			for di in range(-1,2):

				for dj in range(-1,2):

					if(i+di>=0 and i+di<self.N and j+dj>=0 and j+dj<self.M and np.abs(di)+np.abs(dj)==1):

						if self.maze[i+di, j+dj] and dist[i+di, j+dj]>dist[i, j]+1:

							dist[i+di, j+dj]=dist[i, j]+1
							q.put((i+di, j+dj))

		return dist[ei, ej]

class UniformRandomModel:

	def __init__(self):

		self.model={}

	def add(self, x, y, a, nx, ny, r):

		self.model[(x,y,a)]=(nx, ny, r)

	def get_random(self):

		index=np.random.randint(len(self.model.keys()))
		key=list(self.model.keys())[index]

		nx, ny, r=self.model[key]

		return key, nx, ny, r

class PrioritizedSweepingModel:

	def __init__(self):

		self.reverse_model={}
		self.model={}
		self.priority={}
		self.q=[]
		self.theta=0.0001
		heapq.heapify(self.q)

	def addModel(self, x, y, a, nx, ny, r):

		self.model[(x,y,a)]=(nx, ny, r)
		
		if (nx, ny) not in self.reverse_model.keys():

			self.reverse_model[(nx, ny)]=[]
		
		if (x, y, a, r) not in self.reverse_model[(nx,ny)]:
		
			self.reverse_model[(nx,ny)].append((x, y, a, r))

	def add(self, x, y, a, p):

		self.priority[(x, y, a)]=-p
		heapq.heappush(self.q, [-p, (x, y, a)])

	def get(self):

		_p, _key=(None, (None, None, None))

		while not self.empty():

			p, key=heapq.heappop(self.q)
			
			if self.priority[key]==p:
				
				_p, _key=p, key
				self.priority[key]=0

				break

		return _p, _key

	def empty(self):

		if len(self.q)==0:
			
			return True

		return False

class DynaQ:

	def __init__(self, _n, _N, _M, _model):

		self.n=_n
		self.model=_model
		self.Qvalue=np.zeros((_N, _M, len(actions)),dtype='float')
		self.planning_steps=0

	def getAction(self, x, y):

		if np.random.rand()<=eps:

			return np.random.randint(len(actions))

		else:

			return np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[x, y] == np.max(self.Qvalue[x, y]))))

	def performPlanning(self):

		for i in range(self.n):

			self.planning_steps+=1
			key, nx, ny, r=self.model.get_random()
			x, y, a=key

			self.Qvalue[x, y, a]+=alpha*(r+gamma*np.amax(self.Qvalue[nx, ny])-self.Qvalue[x, y, a])

	def getOptimalSteps(self, maze, lim):

		currentX, currentY=maze.start
		steps=0

		while (currentX, currentY) != maze.target and steps<lim:

			currentA=0
				
			currentA=np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[currentX, currentY] == np.max(self.Qvalue[currentX, currentY]))))

			newX, newY=maze.getNext(currentX, currentY, currentA)
			
			steps+=1

			if (newX, newY) == maze.target:
			
				break

			(currentX, currentY)=(newX, newY)

		return steps

	def work(self, maze):

		minimum_steps=maze.getMinSteps()

		while self.getOptimalSteps(maze, minimum_steps+15)-(maze.resolution*4)//2>minimum_steps:

			currentX, currentY=maze.start

			while (currentX, currentY) != maze.target:

				currentA=0
					
				currentA=self.getAction(currentX, currentY)

				newX, newY=maze.getNext(currentX, currentY, currentA)
				
				reward=0
				if (newX, newY) == maze.target:

					reward=1

				self.model.add(currentX, currentY, currentA, newX, newY, reward)

				self.Qvalue[currentX, currentY, currentA]+=alpha*(reward+gamma*np.amax(self.Qvalue[newX, newY])-self.Qvalue[currentX, currentY, currentA])

				(currentX, currentY)=(newX, newY)

				if self.n>0:
					
					self.performPlanning()

		return self.planning_steps

class PrioritizedSweeping:

	def __init__(self, _n, _N, _M, _model):

		self.n=_n
		self.model=_model
		self.Qvalue=np.zeros((_N, _M, len(actions)),dtype='float')
		self.planning_steps=0

	def getAction(self, x, y):

		if np.random.rand()<=eps:

			return np.random.randint(len(actions))

		else:

			return np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[x, y] == np.max(self.Qvalue[x, y]))))

	def performPlanning(self):

		for i in range(self.n):

			p, (x, y, a)=self.model.get()

			if p==None:

				break

			self.planning_steps+=1

	
			nx, ny, r=self.model.model[x, y, a]

			self.Qvalue[x, y, a]+=alpha*(r+gamma*np.amax(self.Qvalue[nx, ny])-self.Qvalue[x, y, a])

			for _x, _y, _a, _r in self.model.reverse_model[(x, y)]:

				_p=np.abs(r+gamma*np.amax(self.Qvalue[x, y])-self.Qvalue[_x, _y, _a])

				if _p>self.model.theta:

					self.model.add(_x, _y, _a, _p)

	def getOptimalSteps(self, maze, lim):

		currentX, currentY=maze.start
		steps=0

		while (currentX, currentY) != maze.target and steps<lim:

			currentA=0
				
			currentA=np.random.choice(np.ndarray.flatten(np.argwhere(self.Qvalue[currentX, currentY] == np.max(self.Qvalue[currentX, currentY]))))

			newX, newY=maze.getNext(currentX, currentY, currentA)
			
			steps+=1

			if (newX, newY) == maze.target:
			
				break

			(currentX, currentY)=(newX, newY)

		return steps

	def work(self, maze):

		minimum_steps=maze.getMinSteps()

		while self.getOptimalSteps(maze, minimum_steps+15)-(maze.resolution*4)//2>minimum_steps:

			currentX, currentY=maze.start

			while (currentX, currentY) != maze.target:

				currentA=0
					
				currentA=self.getAction(currentX, currentY)

				newX, newY=maze.getNext(currentX, currentY, currentA)
				
				reward=0
				if (newX, newY) == maze.target:

					reward=1

				self.model.addModel(currentX, currentY, currentA, newX, newY, reward)

				p=np.abs(reward+gamma*np.amax(self.Qvalue[newX, newY])-self.Qvalue[currentX, currentY, currentA])

				if p>self.model.theta:

					self.model.add(currentX, currentY, currentA, p)

				(currentX, currentY)=(newX, newY)

				if self.n>0:
					
					self.performPlanning()

		return self.planning_steps

def Main():

	mazes=[Maze() for i in range(5)]

	for i in range(5):

		mazes[i].Magnify(i+1)

	runs=50

	stepsPriority=np.zeros(5)
	stepsDyna=np.zeros(5)

	for run in range(runs):

		print('Run: {}'.format(run+1))

		for i, maze in tqdm(enumerate(mazes)):

			model=PrioritizedSweepingModel()
			priority=PrioritizedSweeping(5, maze.N, maze.M, model)

			modelUniform=UniformRandomModel()
			dyna=DynaQ(5, maze.N, maze.M, modelUniform)

			stepsDyna[i]+=dyna.work(maze)
			stepsPriority[i]+=priority.work(maze)

	stepsPriority/=runs
	stepsDyna/=runs

	X=[47*i for i in range(1,6)]

	plt.plot(X, stepsDyna, label='Dyna-Q')
	plt.plot(X, stepsPriority, label='Prioritized Sweeping')

	plt.xticks(X)
	plt.xlabel('Grid Size (# of States)')
	plt.ylabel('Updates till optimality')
	plt.title('Dyna-Q vs Prioritized Sweeping')

	plt.legend()

	plt.savefig('example_8_4.png')
	plt.close()

Main()