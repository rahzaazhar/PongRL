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
	return torch.from_numpy(I).type(torch.FloatTensor)

def batchupdate(prob,returns):

	policy_loss=[]
	returns = torch.tensor(returns)
	for log_prob, R in zip(prob,returns):
		policy_loss.append(-log_prob*R)
	optimizer.zero_grad()
	loss = torch.cat(policy_loss).sum()
	policy_loss.clear()
	print("loss:",loss)
	loss.backward()
	optimizer.step()

def init_weights(model):
	for parameter in model.parameters():
		if parameter.ndimension() == 2:
			torch.nn.init.xavier_uniform(parameter, gain=0.01)

def PolicyGradient(Total_episodes,batch_size,discount_factor):

	render= False
	total_reward=0
	env = gym.make("Pong-v0")
	observation = env.reset()
	prev_x = None# used in computing the difference frame
	running_reward = None
	reward_sum = 0
	episode_number = 0
	prob=[]
	eprew=[]
	reward=0
	returns = []
	while True:
		if render: env.render()
		# preprocess the observation, set input to network to be difference image
		cur_x = prepro(observation)
		#print("obtained preprocessed image")
		x = cur_x - prev_x if prev_x is not None else torch.zeros(1,1,80,80)
		prev_x = cur_x
		aprob = agent.Policypred(x)
		#print("obtained action probability")
		m = Categorical(aprob)
		action = m.sample()
		actionindex = action.item()
		sampled_action = 2 if actionindex==0 else 3 #2 upward 3 downward 
		observation, reward, done , _ = env.step(sampled_action)
		prob.append(m.log_prob(action))
		eprew.append(reward)
		if reward==1 or reward == -1:
			total_reward+=reward
			eprew = discount_rewards(eprew,discount_factor)
			eprew -= np.mean(eprew)
			eprew /= np.std(eprew)
			eprew = eprew.tolist()
			returns.extend(eprew)
			eprew.clear()
			print(eprew)
			print("episode {} ended with reward:{}".format(episode_number+1,reward))
			print("Total reward upto episode {}:{}".format(episode_number+1,total_reward))
			episode_number+=1
			observation = env.reset()
			if episode_number%batch_size==0 and episode_number!=0:
				print("Start Batch Update")
				#print(returns)
				#print(len(prob),len(returns))
				batchupdate(prob,returns)
				prob.clear()
				returns.clear()
		if episode_number==Total_episodes:
			break

max_episodes=1000
batch_size=10
discount=0.99
print("enter main")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = PolicyNN()
init_weights(agent)
#print(agent)
#params = list(agent.parameters())
optimizer = optim.RMSprop(agent.parameters(),lr=0.01,weight_decay=0.99)
PolicyGradient(max_episodes,batch_size,discount)			
			

