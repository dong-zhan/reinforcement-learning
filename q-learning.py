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
def getTargetPosAndInitReward(initPos, action):
	reward = rewards[initPos[0], initPos[1]]
	targetPos = (initPos[0] + action[0], initPos[1] + action[1])
	return targetPos, reward
	
def getTargetPos(initPos, action):
	return (initPos[0] + action[0], initPos[1] + action[1])

def getReward(initPos):
	return rewards[initPos[0], initPos[1]]

def swapValueMaps():
	global valueMap, valueMapNext
	tmpMap = valueMapNext
	valueMapNext = valueMap
	valueMap = tmpMap
	
def init():
	global discountRate, learningRate, gridSizeX, gridSizeY, actions, numIterations, weight, rewards, valueMap, valueMapNext, states	
	discountRate = 0.9 # discounting rate
	learningRate = 0.1
	gridSizeX = 3
	gridSizeY = 2
	actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])	
	numIterations = 1
	weight = 1./len(actions)
	
	rewards = np.full((gridSizeY+2, gridSizeX+2), 0, dtype=float)  
	rewards[1] = [0, 1, 0, 10, 0]
	rewards[2] = [0, -10, 1, 0, 0]
	copyArrayContentToBorder(rewards)

	#make the table 2 units bigger to avoid boundary check, every iteration, copy the content from appropriate row/columns
	valueMap = np.zeros((gridSizeY+2, gridSizeX+2))
	valueMapNext = np.zeros((gridSizeY+2, gridSizeX+2))
	states = [[y, x] for y in range(1,gridSizeY+1) for x in range(1,gridSizeX+1)]		
	
	
def PolicyEvaluation():
	global valueMap, copyValueMap, weight
	for it in range(numIterations):
		for state in states:
			maxQ = -99999
			for action in actions:
				targetPos = getTargetPos(np.array(state), action)
				maxQ = max(maxQ, valueMap[targetPos[0], targetPos[1]])

			currentQ = valueMap[state[0], state[1]]
			dQ = discountRate * maxQ - currentQ
			
			valueMapNext[state[0], state[1]] = currentQ + learningRate * (getReward(state) + dQ)
			
		copyArrayContentToBorder(valueMapNext)
		swapValueMaps()
		
	print(valueMap)
