import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

episodes=200000
actions=[(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
maxVelocity=5
minVelocity=0
gamma=0.75
eps=0.1

def generateRacetrack(n, m):

	track=np.zeros((n,m))
	startLine=[]
	finishLine=[]

	height=np.random.randint(10,21)
	width=np.random.randint(35,46)

	for i in range(height):

		for j in range(width):

			track[i, m-1-j]=1

	width_1=np.random.randint(10,21)
	height_1=np.random.randint(35,51)

	for i in range(height_1):

		for j in range(width_1):

			track[i, m-1-width-j]=1

	height_2=np.random.randint(10,21)

	for i in range(height_2):

		for j in range(m-width):

			track[i+height_1, j]=1

	width_2=np.random.randint(10,21)

	for i in range(height_2+height_1, n):

		for j in range(width_2):

			track[i, j]=1

	for i in range(m):

		if track[n-1, i]:

			track[n-1, i]=2
			startLine.append((n-1, i))

	for i in range(n):

		if track[i, m-1]:

			track[i, m-1]=3
			finishLine.append((i, m-1))

	return track, startLine, finishLine


class RaceTrack:

	def __init__(self, _n, _m):
		
		self.n=_n
		self.m=_m
		self.track, self.startLine, self.finishLine=generateRacetrack(_n, _m)
		self.Qvalue=np.zeros((self.n, self.m, maxVelocity+1, maxVelocity+1, 9))
		self.Cvalue=np.zeros((self.n, self.m, maxVelocity+1, maxVelocity+1, 9))
		self.pi=np.zeros((self.n, self.m, maxVelocity+1, maxVelocity+1), dtype='int')

		for i in range(self.n):
			for j in range(self.m):
				for k in range(maxVelocity+1):
					for l in range(maxVelocity+1):
						for h in range(9):
							
							di, dj=actions[h]

							if k+di>=0 and k+di<=maxVelocity and l+dj>=0 and l+dj<=maxVelocity and k+l+di+dj>0:
								
								self.Qvalue[i,j,k,l,h]=np.random.rand()*10-20

							else:

								self.Qvalue[i,j,k,l,h]=-np.inf

						self.pi[i, j, k, l]=np.argmax(self.Qvalue[i, j, k, l])

	def isOnTrack(self, state):

		x,y=state

		if x<0 or x>=self.n or y<0 or y>=self.m:
			return False

		if self.track[x, y]==0:
			return False

		return True

	def isFinish(self, state):

		x,y=state
		x=max(x,0)
		y=min(y,self.m-1)

		if (x, y) in self.finishLine:
			return True

		return False

	def getStartState(self):

		return self.startLine[np.random.randint(len(self.startLine))], (0,0)

	def getAction(self, position, velocity, policy):

		x,y=position
		i,j=velocity
		possibleActions=[]

		for k in range(9):

			di, dj=actions[k]

			if i+di>=0 and i+di<=maxVelocity and j+dj>=0 and j+dj<=maxVelocity and i+j+di+dj>0:

				possibleActions.append(k)

		possibleActions=np.array(possibleActions)

		action=0

		if policy=='behaviour':

			if np.random.rand()>eps:

				action=self.pi[x, y, i, j]

			else:

				action=np.random.choice(possibleActions)

			if action==self.pi[x, y, i, j]:

				return action, 1-eps+eps/len(possibleActions)

			else:

				return action, eps/len(possibleActions)

		else:

			return self.pi[x, y, i, j], 1		

	def performAction(self, position, velocity, action):

		vx,vy=velocity
		x,y=position
		dvx,dvy=actions[action]

		(nx,ny)=(x-vx,y+vy)
		(nvx,nvy)=(vx+dvx,vy+dvy)

		reward=-1

		if self.isFinish((nx,ny)):

			return (nx, ny), (nvx, nvy), reward

		if not self.isOnTrack((nx,ny)):

			(nx, ny), (nvx, nvy)=self.getStartState()
			reward=-100

		return (nx, ny), (nvx, nvy), reward


	def generateEpisode(self):

		episode=[]
		position, velocity=self.getStartState()
		
		x,y=position
		vx,vy=velocity

		while not self.isFinish(position):

			action, b=self.getAction(position, velocity, 'behaviour')
			newPosition, newVelocity, reward=self.performAction(position, velocity, action)
			episode.append((position, velocity, action, reward, b))

			position=newPosition
			velocity=newVelocity

		return episode


	def monteCarloOffPolicy(self):

		for i in tqdm(range(episodes)):

			episode=self.generateEpisode()
			episode.reverse()

			G=0
			W=1

			for position, velocity, action, reward, b in episode:
				
				x,y=position
				vx,vy=velocity

				G=G*gamma+reward
				self.Cvalue[x, y, vx, vy, action]+=W
				self.Qvalue[x, y, vx, vy, action]+=(W*(G-self.Qvalue[x, y, vx, vy, action]))/self.Cvalue[x, y, vx, vy, action]
				self.pi[x, y, vx, vy]=np.argmax(self.Qvalue[x, y, vx, vy])

				if action!=self.pi[x, y, vx, vy]:
					break

				W=W/b

	def getOptimalTrajectory(self, position, velocity):
		
		trajectory=np.copy(self.track)

		x,y=position
		vx,vy=velocity
		path=[]

		print('-----------')

		while not self.isFinish(position):

			path.append(position)

			action, b=self.getAction(position, velocity, 'target')

			print(action, position)

			position, velocity, _=self.performAction(position, velocity, action)

		x,y=position
		x=max(x,0)
		y=min(y,self.m-1)

		path.append((x,y))

		for i,j in path:
			trajectory[i,j]=4

		path.clear()

		return trajectory

def Main():

	m=200
	n=100

	racetrack=RaceTrack(n, m)

	plt.matshow(racetrack.track)
	plt.title('Generated Track')
	plt.xticks(np.arange(m),['']*m)
	plt.yticks(np.arange(n),['']*n)
	plt.savefig('track.png')
	plt.close()

	racetrack.monteCarloOffPolicy()

	for i,position in enumerate(racetrack.startLine):

		path=racetrack.getOptimalTrajectory(position,(0,0))

		plt.matshow(path)
		plt.title('Optimal Path')
		plt.xticks(np.arange(m),['']*m)
		plt.yticks(np.arange(n),['']*n)
		plt.savefig('path_{}.png'.format(i))
		plt.close()


Main()