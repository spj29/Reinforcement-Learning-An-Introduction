import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d

def GenerateCard():

	return min(np.random.randint(1,14),10)

def GenerateEpisode():

	player_sum=0
	dealer_sum=0
	player_usable_ace=0
	dealer_usable_ace=0
	episode=[]

	dealer_card_1=GenerateCard()
	dealer_card_2=GenerateCard()

	dealer_sum=dealer_card_1+dealer_card_2
	
	if (dealer_card_1==1 or dealer_card_2==1) and dealer_sum<=11:
		dealer_sum+=10
		dealer_usable_ace=1

	while player_sum<12:

		card=GenerateCard()
		player_sum+=card

		if card==1 and player_sum<=11:
			player_sum+=10
			player_usable_ace=1

	while True:

		if player_sum>=20:
			episode.append((player_usable_ace,player_sum,dealer_card_1))
			break

		episode.append((player_usable_ace,player_sum,dealer_card_1))

		card=GenerateCard()
		player_sum+=card

		if player_sum>21 and player_usable_ace:
			player_usable_ace=0
			player_sum-=10

		if player_sum>21:
			return -1, episode

	while True:

		if dealer_sum>=17:
			break

		card=GenerateCard()
		dealer_sum+=card

		if dealer_sum>21 and dealer_usable_ace:
			dealer_usable_ace=0
			dealer_sum-=10

		if dealer_sum>21:
			return 1, episode

	if player_sum>dealer_sum:
		return 1, episode
	elif player_sum==dealer_sum:
		return 0, episode
	else:
		return -1, episode

def Monte_Carlo(episodes):

	StateValue=np.zeros((2,10,10))
	Statecount=np.ones((2,10,10))

	for i in tqdm(range(0,episodes)):

		reward, sequence=GenerateEpisode()

		for player_usable_ace, player_sum, dealer_card_1 in sequence:

			Statecount[player_usable_ace][player_sum-12][dealer_card_1-1]+=1
			StateValue[player_usable_ace][player_sum-12][dealer_card_1-1]+=reward

	StateValue/=Statecount

	return StateValue

def Plot(value, name, title):

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X=np.arange(0,10)
	Y=np.arange(0,10)
	X,Y=np.meshgrid(X,Y)
	
	x_scale=0.75
	y_scale=1
	z_scale=0.25

	scale=np.diag([x_scale, y_scale, z_scale, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=1.0

	def short_proj():
		return np.dot(Axes3D.get_proj(ax), scale)

	ax.get_proj=short_proj

	ax.plot_wireframe(X, Y, value, rstride=1, cstride=1)
	ax.set_xlabel("Dealer Showing")
	ax.set_ylabel("Player Sum")
	ax.set_zlabel("State Value")
	ax.xaxis.labelpad = 20
	ax.yaxis.labelpad = 20
	ax.zaxis.labelpad = 10
	ax.set_xticks([i for i in range(1,11)])
	ax.set_yticks([i for i in range(1,11)])
	ax.set_zticks([i for i in range(-1,2)])
	ax.set_yticklabels([str(i) for i in range(12,22)])
	ax.set_title(title)
	ax.set_zlim(-1,1)
	plt.savefig(name)
	plt.close()

def Main():

	StateValue_10000_episodes=Monte_Carlo(10000)
	StateValue_500000_episodes=Monte_Carlo(500000)

	Plot(StateValue_10000_episodes[0], "No_Usable_Ace_10.png" ,"State Values with No Usable Ace, 10,000 Episodes")
	Plot(StateValue_10000_episodes[1], "Usable_Ace_10.png", "State Values with Usable Ace, 10,000 Episodes")

	Plot(StateValue_500000_episodes[0], "No_Usable_Ace_500.png", "State Values with No Usable Ace, 500,000 Episodes")
	Plot(StateValue_500000_episodes[1], "Usable_Ace_500.png", "State Values with Usable Ace, 500,000 Episodes")

Main()