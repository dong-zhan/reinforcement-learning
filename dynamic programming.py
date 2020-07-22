#reference: https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b

#this is a good practise of dynamic programming - value iteration
#this implementation is just a simple extension of the one in the reference.
#in this implementation, the reward can be dynamic!

def imps():
	global np, random
	import numpy as np
	import random
		
def Parameters():
	global gamma, gridSizeX, gridSizeY, gridSizeX_1, gridSizeY_1, actions, numIterations, weight, rewardPos, rewards, defaultReward
	gamma = 1 # discounting rate
	gridSizeX = 4
	gridSizeY = 4
	gridSizeY_1 = gridSizeY - 1
	gridSizeX_1 = gridSizeX - 1
	actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])	
	numIterations = 1000
	weight = 1./len(actions)
	
	defaultReward = -1
	rewardPos = np.array([(0,0), (3,3)])
	rewards = np.array([0, 0])
	
def GetIndexInRewardPos(pos):
	global rewardPos
	for i, p in enumerate(rewardPos):
		comparison = pos == p
		if comparison.all():
			return i
	return -1
	
def getTargetPosAndReward(initPos, action):
	global rewards, gridSizeX_1, gridSizeY_1
	idx = GetIndexInRewardPos(initPos)
	if not idx == -1:
		return initPos, rewards[idx]
			
	return np.clip(initPos + action, [0,0], [gridSizeY_1, gridSizeX_1]), defaultReward

def Initialization():
	global valueMap, valueMapNext, gridSizeY, gridSizeX, states		#pingpong between valueMap and valueMapNext
	valueMap = np.zeros((gridSizeY, gridSizeX))
	valueMapNext = np.zeros((gridSizeY, gridSizeX))
	states = [[y, x] for y in range(gridSizeY) for x in range(gridSizeX)]	
	
def swapValueMaps():
	global valueMap, valueMapNext
	tmpMap = valueMapNext
	valueMapNext = valueMap
	valueMap = tmpMap
	
def PolicyEvaluation():
	global valueMap, copyValueMap, weight
	for it in range(numIterations):
		for state in states:
			weightedRewards = 0
			for action in actions:
				targetPos, reward = getTargetPosAndReward(np.array(state), action)
				weightedRewards += weight*(reward+(gamma*valueMap[targetPos[0], targetPos[1]])) 				
			valueMapNext[state[0], state[1]] = weightedRewards
			
		swapValueMaps()
