import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

episodes=5000
eps=0.1
alpha=0.1

def windyGridWorld(n, m, wind, start, destination, dx, dy):

	Qvalue=np.zeros((n,m,len(dx)))

	TimeVsEpisodes=[]

	for episode in tqdm(range(episodes)):

		cx, cy=start
		ex, ey=destination

		action=0
		if np.random.rand()<=eps:
			action=np.random.randint(0,len(dx))
		else:
			action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[cx, cy, :] == np.amax(Qvalue[cx, cy, :]))))

		while cx!=ex or cy!=ey:

			nx=cx+dx[action]-wind[cy]
			ny=cy+dy[action]

			nx=min(nx,n-1)
			nx=max(nx,0)

			ny=min(ny,m-1)
			ny=max(ny,0)

			nxt_action=0

			if np.random.rand()<=eps:
				nxt_action=np.random.randint(0,len(dx))
			else:
				nxt_action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[nx, ny, :] == np.amax(Qvalue[nx, ny, :]))))

			Qvalue[cx, cy, action]+=alpha*(-1+Qvalue[nx, ny, nxt_action]-Qvalue[cx, cy, action])

			cx=nx
			cy=ny
			action=nxt_action

			TimeVsEpisodes.append(episode+1)

	return Qvalue, TimeVsEpisodes

def getPath(n, m, wind, start, destination, dx, dy, Qvalue):

	datax=[]
	datay=[]

	cx, cy=start
	ex, ey=destination

	while cx!=ex or cy!=ey:

		datax.append(cy)
		datay.append(7-cx)

		action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[cx, cy, :] == np.amax(Qvalue[cx, cy, :]))))
		nx=cx+dx[action]-wind[cy]
		ny=cy+dy[action]

		nx=min(nx,n-1)
		nx=max(nx,0)

		ny=min(ny,m-1)
		ny=max(ny,0)

		cx=nx
		cy=ny

	datax.append(cy)
	datay.append(7-cx)

	return datax,datay

def KingMoves(dx, dy, name1, name2):

	wind=np.array([0, -2, -3, 5, 1, 1, 2, 2, 1, 0])
	start=(3,0)
	destination=(1,3)
	cx, cy=start
	ex, ey=destination
	n=7
	m=10

	Qvalue, TimeVsEpisodes=windyGridWorld(n, m, wind, start, destination, dx, dy)
	datax, datay=getPath(n, m, wind, start, destination, dx, dy, Qvalue)

	plt.plot(datax, datay)
	sx, sy=start
	plt.scatter(sy,sx+1,s=100,marker='s',color='g')
	plt.scatter(ey,n-ex,s=100,marker='s',color='k')
	plt.title('Greedy Path after {} episodes'.format(episodes))
	plt.xlim(-1, 10)
	plt.ylim(0, 8)
	plt.xticks(np.arange(-1,11), np.append([0], np.append(wind, [0])))
	plt.yticks(np.arange(0,9), ['']*9)
	plt.xlabel('Wind strength')
	plt.grid()
	plt.savefig(name2)
	plt.close()

	return TimeVsEpisodes

def Main():
	
	dx=[-1, 0, 1, 0]
	dy=[0, -1, 0, 1]
	
	elephant=KingMoves(dx, dy, 'TimeVsEpisodes.png', 'OptimalPath_standard.png')

	dx=[-1, 0, 1, 0, 1, -1, 1, -1]
	dy=[0, -1, 0, 1, -1, 1, 1, -1]
	
	king=KingMoves(dx, dy, 'TimeVsEpisodes.png', 'OptimalPath_King.png')

	dx=[-1, 0, 1, 0, 1, -1, 1, -1, 0]
	dy=[0, -1, 0, 1, -1, 1, 1, -1,0]
	
	bonus=KingMoves(dx, dy, 'TimeVsEpisodes.png', 'OptimalPath_Bonus.png')

	plt.plot(elephant, label='Standard 4 Moves')
	plt.plot(king, label='King Moves')
	plt.plot(bonus, label='King Moves+Bonus Move')

	plt.xlabel('Time')
	plt.ylabel('Episode')
	plt.legend()

	plt.savefig('TimeVsEpisodes.png')
	plt.close()

Main()