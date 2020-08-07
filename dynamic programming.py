#reference: https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b

def imps():
	global np, random
	import numpy as np
	import random
	
def swapColumn(arr, c0, c1):
	temp = np.copy(arr[:, c0])
	arr[:, c0] = arr[:, c1]
	my_array[:, c1] = temp
	
def copyColumn(arr, to, fr):
	arr[:, to] = arr[:, fr]
	
def copyRow(arr, to, fr):
	arr[to] = arr[fr]
	
def copyArrayContentToBorder(arr):
	copyColumn(arr, 0, 1)
	copyColumn(arr, gridSizeX+1, gridSizeX)
	copyRow(arr, 0, 1)
	copyRow(arr, gridSizeY+1, gridSizeY)
	
def fillArray(arr):		#for test
	cnt = arr.shape[0] * arr.shape[1]
	arr1 = arr.reshape(cnt)
	for i in range(cnt):
		arr1[i] = i

#reward is from initPos.
def getTargetPosAndReward(initPos, action):
	targetPos = (initPos[0], initPos[1])
	reward = rewards[targetPos]
	if reward == 0:
		return targetPos, reward

	targetPos = (initPos[0] + action[0], initPos[1] + action[1])
	#reward = rewards[targetPos]
	return targetPos, reward

def swapValueMaps():
	global valueMap, valueMapNext
	tmpMap = valueMapNext
	valueMapNext = valueMap
	valueMap = tmpMap
	
def init():
	global gamma, gridSizeX, gridSizeY, actions, numIterations, weight, rewards, valueMap, valueMapNext, states	
	gamma = 1 # discounting rate
	gridSizeX = 4
	gridSizeY = 4
	actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])	
	numIterations = 1
	weight = 1./len(actions)
	
	rewards = np.full((gridSizeY+2, gridSizeX+2), -1, dtype=float)  
	rewards[1,1] = 0
	rewards[4,4] = 0
	copyArrayContentToBorder(rewards)

	#make this 2 units bigger to avoid boundary check, every iteration, copy the content from appropriate row/columns
	valueMap = np.zeros((gridSizeY+2, gridSizeX+2))
	valueMapNext = np.zeros((gridSizeY+2, gridSizeX+2))
	states = [[y, x] for y in range(1,gridSizeY+1) for x in range(1,gridSizeX+1)]		
	
def PolicyEvaluation():
	global valueMap, copyValueMap, weight
	for it in range(numIterations):
		for state in states:
			#print(state, rewards[state[0], state[1]])
			weightedRewards = 0
			for action in actions:
				targetPos, reward = getTargetPosAndReward(np.array(state), action)
				#print(targetPos, reward)
				weightedRewards += weight*(reward+(gamma*valueMap[targetPos[0], targetPos[1]])) 				
			valueMapNext[state[0], state[1]] = weightedRewards
			
		copyArrayContentToBorder(valueMapNext)
		swapValueMaps()
	
	print(valueMap)
			
