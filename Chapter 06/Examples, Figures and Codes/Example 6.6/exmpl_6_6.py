import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dx=[-1, 0, 1, 0]
dy=[0, -1, 0, 1]

def WindyGridWorld_SARSA(n, m, start, destination):

	Qvalue=np.zeros((n,m,4))

	episodes=500
	eps=0.1
	alpha=0.5

	rewards=np.zeros(episodes)

	for episode in range(episodes):

		cx, cy=start
		ex, ey=destination

		action=0
		if np.random.rand()<=eps:
			action=np.random.randint(0,4)
		else:
			action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[cx, cy, :] == np.amax(Qvalue[cx, cy, :]))))

		while cx!=ex or cy!=ey:

			nx=cx+dx[action]
			ny=cy+dy[action]

			nx=min(nx,n-1)
			nx=max(nx,0)

			ny=min(ny,m-1)
			ny=max(ny,0)

			reward=-1

			if nx==n-1 and ny>0 and ny<m-1:
				reward=-100
				nx=n-1
				ny=0

			rewards[episode]+=reward

			nxt_action=0

			if np.random.rand()<=eps:
				nxt_action=np.random.randint(0,4)
			else:
				nxt_action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[nx, ny, :] == np.amax(Qvalue[nx, ny, :]))))

			Qvalue[cx, cy, action]+=alpha*(reward+Qvalue[nx, ny, nxt_action]-Qvalue[cx, cy, action])

			cx=nx
			cy=ny
			action=nxt_action

	return Qvalue, rewards

def WindyGridWorld_Q(n, m, start, destination):

	Qvalue=np.zeros((n,m,4))

	episodes=500
	eps=0.1
	alpha=0.5

	rewards=np.zeros(episodes)

	for episode in range(episodes):

		cx, cy=start
		ex, ey=destination

		while cx!=ex or cy!=ey:

			action=0
			if np.random.rand()<=eps:
				action=np.random.randint(0,4)
			else:
				action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[cx, cy, :] == np.amax(Qvalue[cx, cy, :]))))

			nx=cx+dx[action]
			ny=cy+dy[action]

			nx=min(nx,n-1)
			nx=max(nx,0)

			ny=min(ny,m-1)
			ny=max(ny,0)

			reward=-1

			if nx==n-1 and ny>0 and ny<m-1:
				reward=-100
				nx=n-1
				ny=0

			rewards[episode]+=reward

			Qvalue[cx, cy, action]+=alpha*(reward+np.amax(Qvalue[nx, ny, :])-Qvalue[cx, cy, action])

			cx=nx
			cy=ny

	return Qvalue, rewards

def GeneratePath(Qvalue, start, destination, n, m):

	datax=[]
	datay=[]

	cx, cy=start
	ex, ey=destination

	while cx!=ex or cy!=ey:

		datax.append(cy)
		datay.append(n-cx)

		action=np.random.choice(np.ndarray.flatten(np.argwhere(Qvalue[cx, cy, :] == np.amax(Qvalue[cx, cy, :]))))
		nx=cx+dx[action]
		ny=cy+dy[action]

		nx=min(nx,n-1)
		nx=max(nx,0)

		ny=min(ny,m-1)
		ny=max(ny,0)

		if nx==n-1 and ny>0 and ny<m-1:
			nx=n-1
			ny=0

		cx=nx
		cy=ny

	datax.append(cy)
	datay.append(n-cx)
	
	return datax, datay

def Main():

	n=4
	m=12
	start=(n-1,0)
	destination=(n-1,m-1)

	cx, cy=start
	ex, ey=destination

	# Plot Paths
	Qvalue_SARSA, rewards_SARSA=WindyGridWorld_SARSA(n, m, start, destination)
	Qvalue_Q, rewards_Q=WindyGridWorld_Q(n, m, start, destination)

	datax_SARSA, datay_SARSA=GeneratePath(Qvalue_SARSA, start, destination, n, m)
	datax_Q, datay_Q=GeneratePath(Qvalue_Q, start, destination, n, m)
	
	plt.plot(datax_SARSA, datay_SARSA,label='SARSA')
	plt.plot(datax_Q, datay_Q, label='Q Learning')

	sx, sy=start
	plt.scatter(sy,n-sx,s=100,marker='s',color='g')
	plt.scatter(ey,n-ex,s=100,marker='s',color='k')
	plt.scatter(np.arange(1,11),np.ones(10)*(n-ex),s=100,marker='s',color='r')
	plt.title('Greedy Paths after 500 episodes')
	plt.xlim(-1, m+1)
	plt.ylim(0, n+1)
	plt.xticks(np.arange(-1, m+1))
	plt.yticks(np.arange(0, n+1))
	plt.grid()
	plt.legend()
	plt.savefig('OptimalPath.png')
	plt.close()

	# Plot Rewards:
	runs=500
	total_rewards_SARSA=np.zeros(500)
	total_rewards_Q=np.zeros(500)

	for run in tqdm(range(runs)):

		_, rewards_SARSA=WindyGridWorld_SARSA(n, m, start, destination)
		_, rewards_Q=WindyGridWorld_Q(n, m, start, destination)

		total_rewards_SARSA+=rewards_SARSA
		total_rewards_Q+=rewards_Q

	total_rewards_SARSA/=runs
	total_rewards_Q/=runs

	plt.plot(total_rewards_SARSA, label='SARSA')
	plt.plot(total_rewards_Q, label='Q Learning')
	plt.ylim(-101,1)
	plt.xlabel('Episode Number')
	plt.ylabel('Reward')
	plt.title('Sum of Rewards Obtained in Episodes')
	plt.legend()
	plt.savefig('Rewards.png')
	plt.close()

Main()