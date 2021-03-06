#reference: https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b

#code is derived from the reference.


def imps():
	global np, random
	import numpy as np
	import random
	
def params():
	global gamma, gridSizeX, gridSizeY, gridSizeY_1, gridSizeX_1
	global actions, numIterations
	global terminationStates, defaultReward, rewards
	
	gamma = 0.6 # discounting rate

	gridSizeX = 4
	gridSizeY = 6
	gridSizeY_1 = gridSizeY - 1
	gridSizeX_1 = gridSizeX - 1

	actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
	numIterations = 1				#set numIterations to 1, and run calc, print(returns, V), and observe what calc is approximating...
	
	defaultReward = -1
	terminationStates = np.array([(0,0), (3,3)])
	rewards = np.array([0, 0])
	
def GetIndexInTerminationStates(pos):
	global terminationStates
	for i, p in enumerate(terminationStates):
		comparison = pos == p
		if comparison.all():
			return i
	return -1
	
def listInListArray(l, la):
	for i, p in enumerate(la):
		comparison = l == p
		if comparison.all():
			return True
	return False
	
def getTargetPosAndReward(initPos, action):
	global rewards, gridSizeX_1, gridSizeY_1
	idx = GetIndexInTerminationStates(initPos)
	if not idx == -1:
		return initPos, rewards[idx]
			
	return np.clip(initPos + action, [0,0], [gridSizeY_1, gridSizeX_1]), defaultReward
	
def init():
	global V, returns, states
	V = np.zeros((gridSizeY, gridSizeX))
	returns = {(y, x):list() for y in range(gridSizeY) for x in range(gridSizeX)}
	states = [[y, x] for y in range(gridSizeY) for x in range(gridSizeX)]

def generateEpisode(initState):	  	#generate episode following PI: S0, A0, R1, S1, 
	global terminationStates
	episode = []
	while True:
		if listInListArray(initState, terminationStates) :
			return episode
		action = random.choice(actions)
		finalState, reward = getTargetPosAndReward(np.array(initState), action)
		episode.append([list(initState), action, reward, list(finalState)])
		initState = finalState
		
def calc():
	global V, returns, states
	for it in range(numIterations):		
		initState = random.choice(states[1:-1])
		episode = generateEpisode(initState)
		G = 0
		for i, step in enumerate(episode[::-1]):		#enumerate in reversed order
			#print(step[0], [x[0] for x in episode[::-1][len(episode)-i:]])
			G = gamma*G + step[2]						#G <--- gamma*G + Rt+1
			if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:		#if St not appears in S0,S1,S2...,St-1 -> reversed order.
				idx = (step[0][0], step[0][1])
				returns[idx].append(G)								#append G to returns
		
	for v in returns:
		if len(returns[v]) > 0 :
			V[v] = np.average(returns[v])		#average(Returns(St)) -> V(St)
		else :
			V[v] = 0

	print(V)
