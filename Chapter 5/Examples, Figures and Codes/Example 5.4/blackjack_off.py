import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns

Policy=np.ones((2,10,10))

for i in range(2):
	for j in range(12,22):
		for k in range(1,11):
			if j>=20:
				Policy[i,j-12,k-1]=0
			else:
				Policy[i,j-12,k-1]=1

def GenerateCard():

	return min(np.random.randint(1,14),10)

def GenerateEpisode(initial_state=[]):

	player_sum=0
	dealer_sum=0
	player_usable_ace=0
	dealer_usable_ace=0
	episode=[]

	dealer_card_1=initial_state[2]
	dealer_card_2=GenerateCard()

	dealer_sum=dealer_card_1+dealer_card_2
	
	if (dealer_card_1==1 or dealer_card_2==1) and dealer_sum<=11:
		dealer_sum+=10
		dealer_usable_ace=1

	player_sum=initial_state[1]
	player_usable_ace=initial_state[0]
	action=0

	while True:

		action=np.random.randint(0,2)

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

def Monte_Carlo_Off_Policy(episodes):

	weights=np.zeros(episodes)
	returns=np.zeros(episodes)

	for i in range(0,episodes):

		returns[i], sequence=GenerateEpisode([1,13,2])
		weight=1

		for (player_usable_ace, player_sum, dealer_card_1),action in sequence:

			weight*=2

			if action != Policy[player_usable_ace, player_sum-12, dealer_card_1-1]:

				weight=0
				break

		weights[i]=weight

	weighted_returns=weights*returns
	weighted_returns=np.add.accumulate(weighted_returns)
	weights=np.add.accumulate(weights)

	ordinary_sampling=weighted_returns/np.arange(1, episodes+1)
	with np.errstate(divide='ignore',invalid='ignore'):
		weighted_sampling=np.where(weights!=0, weighted_returns/weights, 0)

	return ordinary_sampling, weighted_sampling

def Main():

	true_value=-0.27726
	runs=100
	episodes=10000

	ordinary_error=np.zeros(episodes)
	weighted_error=np.zeros(episodes)

	for i in tqdm(range(100)):
		
		ordinary_sampling, weighted_sampling=Monte_Carlo_Off_Policy(episodes)
		ordinary_error+=np.power(ordinary_sampling-true_value,2)
		weighted_error+=np.power(weighted_sampling-true_value,2)

	ordinary_error/=runs
	weighted_error/=runs

	plt.plot(ordinary_error, label='Ordinary Importance Sampling')
	plt.plot(weighted_error, label='Weighted Importance Sampling')
	plt.title('Ordinary vs Weighted Importance Sampling')
	plt.xlabel('Episodes')
	plt.ylabel('Mean Square Error (Over 100 runs)')
	plt.xscale('log')
	plt.legend()
	plt.savefig('meansquareerror.png')
	plt.close()

Main()