import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns

def GenerateCard():

	return min(np.random.randint(1,14),10)

def GenerateEpisode(policy):

	player_sum=0
	dealer_sum=0
	player_usable_ace=0
	dealer_usable_ace=0
	episode=[]

	dealer_card_1=np.random.randint(1,11)
	dealer_card_2=GenerateCard()

	dealer_sum=dealer_card_1+dealer_card_2
	
	if (dealer_card_1==1 or dealer_card_2==1) and dealer_sum<=11:
		dealer_sum+=10
		dealer_usable_ace=1

	player_sum=np.random.randint(12,22)
	player_usable_ace=np.random.randint(0,2)
	action=0
	first=1
	while True:

		if first==0:
			action=policy[player_usable_ace,player_sum-11,dealer_card_1-1]
		else:
			action=np.random.randint(0,2)
			first=0

		if action==0:
			episode.append([(player_usable_ace,player_sum,dealer_card_1),0])
			break

		episode.append([(player_usable_ace,player_sum,dealer_card_1),1])

		card=GenerateCard()
		player_sum+=card

		if player_sum>21 and player_usable_ace>0:
			player_usable_ace=0
			player_sum-=10

		if player_sum>21:
			return -1, episode

	while True:

		if dealer_sum>=17:
			break

		card=GenerateCard()
		dealer_sum+=card

		if dealer_sum>21 and dealer_usable_ace>0:
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

def Monte_Carlo_es(episodes):

	QValue=np.zeros((2,2,10,10))
	Qcount=np.ones((2,2,10,10))

	InitialPolicy=np.ones((2,11,10))
	Policy=np.ones((2,11,10))

	for i in range(2):
		for j in range(12,22):
			for k in range(1,11):
				if j>=20:
					InitialPolicy[i,j-11,k-1]=0
				else:
					InitialPolicy[i,j-11,k-1]=1
				Policy[i,j-11,k-1]=np.random.randint(2)

	for i in tqdm(range(0,episodes)):

		reward, sequence=GenerateEpisode(InitialPolicy if i==0 else Policy)

		for (player_usable_ace, player_sum, dealer_card_1),action in sequence:

			Qcount[action,player_usable_ace,player_sum-12,dealer_card_1-1]+=1
			QValue[action,player_usable_ace,player_sum-12,dealer_card_1-1]+=reward

			stick=QValue[0,player_usable_ace,player_sum-12,dealer_card_1-1]/Qcount[0,player_usable_ace,player_sum-12,dealer_card_1-1]
			hit=QValue[1,player_usable_ace,player_sum-12,dealer_card_1-1]/Qcount[1,player_usable_ace,player_sum-12,dealer_card_1-1]

			if stick>=hit:
				Policy[player_usable_ace,player_sum-11,dealer_card_1-1]=0
			elif stick==hit:
				Policy[player_usable_ace,player_sum-11,dealer_card_1-1]=np.random.randint(0,2)
			else:
				Policy[player_usable_ace,player_sum-11,dealer_card_1-1]=1

	return QValue/Qcount, Policy

def PlotWire(value, title, name):

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
	ax.set_title(title)
	ax.xaxis.labelpad = 20
	ax.yaxis.labelpad = 20
	ax.zaxis.labelpad = 10
	ax.set_xticks([i for i in range(1,11)])
	ax.set_yticks([i for i in range(1,11)])
	ax.set_zticks([i for i in range(-1,2)])
	ax.set_yticklabels([str(i) for i in range(12,22)])
	ax.set_zlim(-1,1)
	plt.savefig(name)
	plt.close()

def PlotHeatMap(value, title, name):

	fig1 = sns.heatmap(np.flipud(value), cmap="YlGnBu",xticklabels=range(1, 11),yticklabels=list(reversed(range(11, 22))))
	fig1.set_ylabel('Player Sum')
	fig1.set_xlabel('Dealer Showing')
	fig1.set_title(title)
	plt.savefig(name)
	plt.close()	

def Main():

	QValue, Policy=Monte_Carlo_es(5000000)

	PlotWire(np.max(QValue[:,0,:,:],axis=0), "State Values with No Usable Ace", "No_Usable_Ace_ES.png")
	PlotWire(np.max(QValue[:,1,:,:],axis=0), "State Values with Usable Ace", "Usable_Ace_ES.png")

	PlotHeatMap(Policy[0], "Optimal Action with No Usable Ace", "action_No_Ace_ES.png")
	PlotHeatMap(Policy[1], "Optimal Action with Usable Ace", "action_With_Ace_ES.png")

Main()