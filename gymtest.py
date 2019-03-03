import gym
env = gym.make('Pong-v0')
for i_episodes in range(10):
	observation = env.reset()
	cnt=0
	totalreward=0
	while True:
		env.render()
		#print(observation)
		action = env.action_space.sample()
		observation, reward, done , info = env.step(action)
		totalreward+=reward
		cnt+=1
		if done:
			print("Episode {} finished after {} timesteps total reward:{}".format(i_episodes+1,cnt,totalreward))
			break
