import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
from torch.distributions import Categorical
from matplotlib import pyplot as plt
import random

class PolicyNN(nn.Module):

	def __init__(self):
		super(PolicyNN,self).__init__()
		#input 1*80*80
		
		self.fc1 = nn.Linear(6400,200)
		self.fc2 = nn.Linear(200,2)


	def Policypred(self, x):
		#block1
		x = x.view(-1, 80*80)
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x),dim=-1)
		return x

	def init_weights(self,initialisation="Xavier"):
	
		if(initialisation=="Xavier"):
			for parameter in self.parameters():
				if parameter.ndimension() == 2:
					torch.nn.init.xavier_uniform(parameter, gain=0.01)
		if(initialisation=="loadweights"):
			self.load_state_dict(torch.load("/home/azhar/projects/pong/weightsAdam1/weightV1.0_trained1200.pt", map_location="cuda:0"))


def discount_rewards(r,gamma):

	discounted_r = []
	running_add = 0
	for t in reversed(range(0, len(r))):
		if r[t] != 0: running_add = 0# reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r.append(running_add)
	discounted_r = reversed(discounted_r)
	return list(discounted_r)

def prepro(I):

	I = I[35:195]
	I = I[::2,::2,0]
	I[I == 144] = 0
	I[I == 109] = 0
	I[I != 0] = 1
	I = np.expand_dims(I,axis=0)
	I = np.expand_dims(I,axis=0)
	return torch.from_numpy(I).type(torch.FloatTensor)

def accumulategrad(prob,returns):

	policy_loss=[]
	loss=0
	returns = torch.tensor(returns)
	returns = returns.to(device)
	for log_prob, R in zip(prob,returns):
		policy_loss.append(-log_prob*R)

	for i in policy_loss:
		loss=loss+i
	policy_loss.clear()
	loss.backward()
	



def plotgraph(reward,episode,rreward):
		plt.clf()
		plt.plot(episode, reward, label="reward per episode")
		plt.legend()
		str1 = '/home/azhar/projects/pong/plots/reward-VS-episodeV1.0new.png'
		fig.savefig(str1)
		plt.clf()
		plt.plot(episode, rreward, label="running reward per episode")
		plt.legend()
		str1 = '/home/azhar/projects/pong/plots/runningreward-VS-episode1_V1.0new.png'
		fig.savefig(str1)



def PolicyGradient(Total_episodes,batch_size,discount_factor,epsilon):

	render= True
	total_reward=0
	env = gym.make("Pong-v0")
	observation = env.reset()
	prev_x = None# used in computing the difference frame
	running_reward = None
	reward_sum = 0
	episode_number = 1100
	prob=[]
	eprew=[]
	reward=0
	pltep=[]
	pltrew=[]
	pltrrew=[]

	while True:

		#if(episode_number%200==0):
		#	epsilon=epsilon/5
		if render: env.render()
		# preprocess the observation, set input to network to be difference image
		cur_x = prepro(observation)
		x = cur_x - prev_x if prev_x is not None else torch.zeros(1,1,80,80)
		x = x.to(device)
		prev_x = cur_x
		aprob = agent.Policypred(x)
		uni_p=np.random.uniform()
		#epsilon-greedy algorithm for exploration and exploitation
		if(uni_p>epsilon):
			k=int(torch.argmax(aprob))
		else:
			k=np.random.choice([0,1])
		#print("random prob:{} prediction:{} index:{}".format(uni_p,aprob,k))
		sampled_action = idx_actionmap[k] #2 upward 3 downward 
		observation, reward, done , _ = env.step(sampled_action)
		if sampled_action==2: # go up
			prob.append(torch.log(aprob[0][0]))
		elif sampled_action==3:# go down
			prob.append(torch.log(aprob[0][1]))
		eprew.append(reward)
		if reward==1 or reward == -1 :
			print("episode {} game ended with reward:{}".format(episode_number+1,reward))
			total_reward+=reward
		
		if done:
			print("Total reward in episode {}:{}".format(episode_number+1,total_reward))
			running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
			observation=env.reset()
			pltrrew.append(running_reward)
			pltrew.append(total_reward)
			total_reward=0
			pltep.append(episode_number)
			episode_number+=1
			plotgraph(pltrew,pltep,pltrrew)
			eprew = discount_rewards(eprew,discount_factor)
			eprew -= np.mean(eprew)
			eprew /= np.std(eprew)
			eprew = eprew.tolist()
			accumulategrad(prob,eprew)
			prob.clear()
			eprew.clear()

			if episode_number%batch_size==0 and episode_number!=0:
				
				print("----------------------------------------accumulated gradients-----------------------------------")
				print("----------------------------------------gradiets for fc1 layer-----------------------------------")
				print(agent.fc1.weight.grad.sum())
				print("----------------------------------------gradiets for fc2 layer-----------------------------------")
				print(agent.fc2.weight.grad.sum())
				print("Start Batch Update")
				print("-----------------------------------------before optimizer step---------------------------------------")
				print("----------------------------------------weights for fc1 layer-----------------------------------")
				print(agent.fc1.weight.sum())
				print("----------------------------------------weights for fc2 layer-----------------------------------")
				print(agent.fc2.weight.sum())
				optimizer.step()
				print("-----------------------------------------after optimizer step---------------------------------------")
				print("----------------------------------------gradiets for fc1 layer-----------------------------------")
				print(agent.fc1.weight.sum())
				print("----------------------------------------gradiets for fc2 layer-----------------------------------")
				print(agent.fc2.weight.sum())
				optimizer.zero_grad()
		if episode_number%100==0 and episode_number!=0:
			#epsilon=epsilon/episode_number
			torch.save(agent.state_dict(),"/home/azhar/projects/pong/weightsAdam1/weightV1.0_trained"+str(episode_number)+".pt")
			

max_episodes=1000
batch_size=2
discount=0.99
epsilon=0.2
fig=plt.figure()
print("enter main")
idx_actionmap={0:2,1:3}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
agent = PolicyNN()
agent.to(device)
agent.init_weights("loadweights")

#agent.load_state_dict(torch.load("/home/azhar/projects/pong/weights/weight600.pt", map_location="cuda:0"))


#print(agent)
#params = list(agent.parameters())
#optimizer = optim.RMSprop(agent.parameters(),lr=0.01,weight_decay=0.99)
learning_rate = 1e-4
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
optimizer.zero_grad()
PolicyGradient(max_episodes,batch_size,discount,0.2)			
			

