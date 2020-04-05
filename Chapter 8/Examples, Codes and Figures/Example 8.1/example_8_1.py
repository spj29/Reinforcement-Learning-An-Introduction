import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

gamma=0.95
episodes=50
alpha=0.1
eps=0.1

actions=[(0, 1), (0, -1), (1, 0), (-1, 0)]

N=6
M=9

maze=np.ones((N, M))
maze[0, 7]=0
maze[1, 7]=0
maze[2, 7]=0
maze[1, 2]=0
maze[2, 2]=0
maze[3, 2]=0
maze[4, 5]=0

target=(0, 8)
start=(2, 0)

def getNext(x, y, a):

	dx, dy=actions[a]

	if x+dx>=N or x+dx<0 or y+dy>=M or y+dy<0:

		return (x,y)

	if maze[x+dx, y+dy]:

		return (x+dx, y+dy)

	return (x,y)


class DynaQ:

	def __init__(self, _n):

		self.n=_n
		self.Qvalue=np.zeros((N, M, len(actions)))
		self.model={}
		self.stepsPerEpisode=np.zeros(episodes)

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

			r, nx, ny=self.model[(x, y, a)]

			self.Qvalue[x, y, a]+=alpha*(r+gamma*np.amax(self.Qvalue[nx, ny])-self.Qvalue[x, y, a])

	def work(self):

		for episode in range(episodes):

			currentX, currentY=start

			steps=0

			while (currentX, currentY) != target:

				currentA=self.getAction(currentX, currentY)
				newX, newY=getNext(currentX, currentY, currentA)

				reward=0
				if (newX, newY) == target:
					reward=1

				self.model[(currentX, currentY, currentA)]=(reward, newX, newY)

				self.Qvalue[currentX, currentY, currentA]+=alpha*(reward+gamma*np.amax(self.Qvalue[newX, newY])-self.Qvalue[currentX, currentY, currentA])

				(currentX, currentY)=(newX, newY)

				if self.n>0:
					
					self.performPlanning()

				steps+=1

			self.stepsPerEpisode[episode]=steps

def Main():

	runs=30

	steps_no_planning=np.zeros(episodes)
	steps_planning_5_steps=np.zeros(episodes)
	steps_planning_50_steps=np.zeros(episodes)

	for run in tqdm(range(runs)):

		no_planning=DynaQ(0)
		planning_5_steps=DynaQ(5)
		planning_50_steps=DynaQ(50)

		no_planning.work()
		planning_5_steps.work()
		planning_50_steps.work()

		steps_no_planning+=no_planning.stepsPerEpisode
		steps_planning_5_steps+=planning_5_steps.stepsPerEpisode
		steps_planning_50_steps+=planning_50_steps.stepsPerEpisode

	steps_no_planning/=runs
	steps_planning_5_steps/=runs
	steps_planning_50_steps/=runs

	plt.plot(steps_no_planning, label='No Planning')
	plt.plot(steps_planning_5_steps, label='Planning with 5 steps')
	plt.plot(steps_planning_50_steps, label='Planning with 50 steps')

	plt.ylim(14, 1000)

	plt.xlabel('Episodes')
	plt.ylabel('Steps Per Episode')
	plt.title('Learning Curves with Different Number of Planning Steps')
	plt.legend()

	plt.savefig('figure_8_2.png')

Main()