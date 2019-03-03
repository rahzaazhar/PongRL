import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch

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

def one_hot_encoding(actionindex):
	if actionindex==0:
		return [1,0]
	else:
		return [0,1]

def discount_rewards(r,gamma):

	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		if r[t] != 0: running_add = 0# reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

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
	print(agent)
	params = list(agent.parameters())
	print(len(params))
	Total_reward=[]
	optimizer = optim.RMSprop(agent.parameters(),lr=0.01,weight_decay=0.99)
	criterion = nn.CrossEntropyLoss()
	render= False
	env = gym.make("Pong-v0")
	observation = env.reset()
	prev_x = None# used in computing the difference frame
	running_reward = None
	reward_sum = 0
	episode_number = 0
	prob=[]
	targets=[]
	eprew=[]
	prob_batch=[]
	targets_batch=[]
	discounted_rew=[]
	for episode_number in range(Total_episodes):
		reward_sum=0
		cnt = 0
		observation = env.reset()
		while True:
			if render: env.render()
			# preprocess the observation, set input to network to be difference image
			cur_x = prepro(observation)
			#print(cur_x.type())
			x = cur_x - prev_x if prev_x is not None else torch.zeros(1,1,80,80)
			x = x.type(torch.FloatTensor)
			#print(x.type())
			prev_x = cur_x
			# forward the policy network and sample an action from the returned probability
			aprob = agent.Policypred(x)
			print(aprob)
			actionindex = int(torch.argmax(aprob))
			target = one_hot_encoding(actionindex)
			sampled_action = 2 if actionindex==0 else 3
			observation, reward, done , info = env.step(sampled_action)
			prob.append(aprob)
			targets.append(target)
			eprew.append(reward)
			reward_sum +=reward
			#print("first time step done")
			cnt+=1
			if done:
				Total_reward.append(reward_sum)
				print("Episode {} finished after {} timesteps total reward:{}".format(episode_number+1,cnt,reward_sum))
				break
		if (episode_number+1)%batch_size==0:
			prob_batch = np.vstack(prob)
			targets_batch = np.vstack(targets)
			discounted_rew = np.vstack(discount_rewards(eprew,discount_factor))
			discounted_rew -= np.mean(discounted_epr)
			discounted_rew /= np.std(discounted_epr)
			optimizer.zero_grad()
			outputs = prob_batch*discounted_rew
			loss = criterion(outputs, targets_batch)
			loss.backward()
			optimizer.step()
			prob=[]
			targets=[]
			eprew=[]
			prob_batch=[]
			targets_batch=[]
			discounted_rew=[]

#if __name__ == "main":
max_episodes=10
batch_size=2
discount=0.99
print("enter main")
PolicyGradient(max_episodes,batch_size,discount)			



