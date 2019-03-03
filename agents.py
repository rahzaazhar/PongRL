import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
from torch.distributions import Categorical


class PolicyNN(nn.Module):

	def __init__(self):
		super(PolicyNN,self).__init__()
		#input 1*80*80
		self.conv1=nn.Conv2d(1,6,5)#6*76*76
		self.conv2=nn.Conv2d(6,6,5)#6*72*72
		self.pool1=nn.MaxPool2d(2,2)#6*36*36
		self.conv3=nn.Conv2d(6,6,5)#6*32*32
		self.conv4=nn.Conv2d(6,6,5)#6*28*28
		self.pool2 = nn.MaxPool2d(2, 2)#6*14*14
		self.conv5=nn.Conv2d(6,16,5)#16*10*10
		self.pool3 = nn.MaxPool2d(2, 2)#16*5*5
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,2)

	def Policypred(self, x):
		#block1
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		#block2
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		#block3
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x),dim=-1)
		return x

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
	return torch.from_numpy(I)


def PolicyGradient(Total_episodes,batch_size,discount_factor):
	print("enter policy")
	agent = PolicyNN()
	#agent = nn.DataParallel(agent)
	print(agent)
	params = list(agent.parameters())
	Total_reward=[]
	optimizer = optim.RMSprop(agent.parameters(),lr=0.01,weight_decay=0.99)
	render= False
	env = gym.make("Pong-v0")
	observation = env.reset()
	prev_x = None# used in computing the difference frame
	running_reward = None
	reward_sum = 0
	episode_number = 0
	prob=[]
	eprew=[]
	policy_loss=[]
	discounted_rew= None
	for episode_number in range(Total_episodes):
		reward_sum=0
		cnt = 0
		observation = env.reset()
		while True:
			if render: env.render()
			# preprocess the observation, set input to network to be difference image
			cur_x = prepro(observation)
			x = cur_x - prev_x if prev_x is not None else torch.zeros(1,1,80,80)
			x = x.type(torch.FloatTensor)
			prev_x = cur_x
			# forward the policy network and sample an action from the returned probability
			aprob = agent.Policypred(x)
			m = Categorical(aprob)
			action = m.sample()
			actionindex = action.item()
			sampled_action = 2 if actionindex==0 else 3 #2 upward 3 downward 
			observation, reward, done , _ = env.step(sampled_action)
			prob.append(m.log_prob(action))
			eprew.append(reward)
			reward_sum +=reward
			cnt+=1
			if done or cnt==500:
				Total_reward.append(reward_sum)
				print("Episode {} finished after {} timesteps total reward:{}".format(episode_number+1,cnt,reward_sum))
				break
		
		discounted_rew = discount_rewards(eprew,discount_factor)
		discounted_rew = list(discounted_rew)
		discounted_rew = torch.tensor(discounted_rew)
		discounted_rew = (discounted_rew - discounted_rew.mean()/(discounted_rew.std()))
		for log_prob, R in zip(prob,discounted_rew):
			policy_loss.append(-log_prob*R)
		optimizer.zero_grad()
		loss = torch.cat(policy_loss).sum()
		loss.backward()
		optimizer.step()
		prob.clear()
		eprew.clear()
		policy_loss.clear()


max_episodes=10
batch_size=2
discount=0.99
print("enter main")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PolicyGradient(max_episodes,batch_size,discount)			



