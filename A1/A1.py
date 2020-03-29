import math
import copy
import random

class GridWorld:
    def __init__(self, state, terminalState, gamma, threshold):
        self.state = state #|S|
        self.terminalState = terminalState
        self.gamma = gamma #gamma
        self.threshold = threshold #theta

        self.value = []
        self.optimalPolicy = []
        self.gridSize = int(math.sqrt(self.state))
        self.action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3} #A
        self.trans = [[0 for j in range(len(self.action))] for i in range(self.state)] #s' = trans[s][a] 
        self.prob = [[[0.0 for k in range(self.state)] for j in range(len(self.action))] for i in range(self.state)] #P[s][a][s']
        self.reward = [[-1.0 for j in range(len(self.action))] for i in range(self.state)] #E[R[s][a]]
        self.policy = [[0.25 for j in range(len(self.action))] for i in range(self.state)] #pi[s][a]
        self.map = [[(j + i * self.gridSize) for j in range(self.gridSize)] for i in range(self.gridSize)]
        self.__calcParam()

    def __calcParam(self):
        curState = 0;
        for i in range(self.gridSize): #row_index
            for j in range(self.gridSize): #col_index
                if i == 0:
                    self.trans[curState][self.action['n']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['n']] = self.map[i-1][j]
                if i == self.gridSize - 1:
                    self.trans[curState][self.action['s']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['s']] = self.map[i+1][j]
                if j == 0:
                    self.trans[curState][self.action['w']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['w']] = self.map[i][j-1]
                if j == self.gridSize - 1:
                    self.trans[curState][self.action['e']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['e']] = self.map[i][j+1]
                curState += 1
        
        for i in range(self.state):
            for a in self.action.keys():
                if i not in self.terminalState:
                    self.prob[i][self.action[a]][self.trans[i][self.action[a]]] = 1.0
                else:
                    self.reward[i][self.action[a]] = 0.0
                    self.policy[i][self.action[a]] = 0.0

    def evaluation(self):
        #Initialize
        self.value = [random.random() for i in range(self.state)]
        for i in self.terminalState:
            self.value[i] = 0.0
        # print(self.value)

        k = 0
        # print("k = 0")
        # self.printGridValue()
        #Loop
        while True:
            delta = 0.0
            for curState in range(self.state):
                oldValue = self.value[curState]
                newValue = 0.0
                for a in self.action.keys(): #GridWorld Specified
                    tmp = 0.0
                    #the inner sum degraded
                    nextState = self.trans[curState][self.action[a]] #Only one element because here's GridWorld
                    tmp += self.prob[curState][self.action[a]][nextState] * self.value[nextState] #prob must be 1.0
                    newValue += self.policy[curState][self.action[a]] * (self.reward[curState][self.action[a]] + (self.gamma * tmp)) 
                self.value[curState] = newValue
                delta = max(delta, math.fabs(oldValue - self.value[curState]))
            k += 1
            # print("k = %d"%(k))
            # print("delta = %0.6f"%(delta))
            # self.printGridValue()
            if delta < self.threshold:
                break
        print(k)

    def policyIteration(self):
        #Initialize
        self.value = [random.random() for i in range(self.state)]
        # for i in self.terminalState:
        #     self.value[i] = 0.0
        # print(self.value)
        self.optimalPolicy = [random.randint(0, 3) for i in range(self.state)]
        #print(self.optimalPolicy)

        numPass = 0
        while True:
            print("pass = %d"%(numPass))
            k = 0
            #print("k = 0")
            self.printGridValue()
            #Loop
            #while True:
            #delta = 0.0
            for curState in range(self.state):
                oldValue = self.value[curState]
                #using determinstic policy
                nextState = self.trans[curState][self.optimalPolicy[curState]] #Only one element because here's GridWorld
                tmp = self.prob[curState][self.optimalPolicy[curState]][nextState] * self.value[nextState] #prob must be 1.0
                newValue = self.reward[curState][self.optimalPolicy[curState]] + (self.gamma * tmp) 
                self.value[curState] = newValue
                #delta = max(delta, math.fabs(oldValue - self.value[curState]))
            #k += 1
            #print("k = %d"%(k))
            #print("delta = %0.6f"%(delta))
            self.printGridValue()
            #if delta < self.threshold or k > 100:
            #    break
            #print(k)

            #Policy Improvement
            policyStable = True
            for curState in range(self.state):
                oldAction = self.optimalPolicy[curState]
                newAction = 0
                maxValue = -1.0e9
                for a in self.action.keys():
                    nextState = self.trans[curState][self.action[a]] #Only one element because here's GridWorld
                    tmp = self.prob[curState][self.action[a]][nextState] * self.value[nextState] #prob must be 1.0
                    tmpValue = self.reward[curState][self.action[a]] + (self.gamma * tmp)
                    if tmpValue > maxValue:
                        newAction = self.action[a]
                        maxValue = tmpValue
                self.optimalPolicy[curState] = newAction
                if oldAction != self.optimalPolicy[curState]:
                    policyStable = False
            self.printOptimalPolicy()
            if policyStable:
                break
            numPass += 1

    def valueIteration(self):
        pass

    def printGridValue(self):
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                curState = j + i * self.gridSize
                print("%0.2f\t"%(self.value[curState]), end='')
            print()

    def printOptimalPolicy(self):
        reverseTable = {0 : 'n', 1 : 'e', 2 : 's', 3 : 'w'}
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                curState = j + i * self.gridSize
                action = reverseTable[self.optimalPolicy[curState]]
                print("%s\t"%(action), end='')
            print()

    def validationTest(self):
        for i in range(self.state):
            for a in self.action.keys():
                for j in range(self.state):
                    if not math.fabs(self.prob[i][self.action[a]][j]) <= 1e-6:
                        print("state %d in action %s to state %d has prob %f"%(i, a, j, self.prob[i][self.action[a]][j]))
            print()

def main():
    gridWorld = GridWorld(state=16, terminalState=[0, 15], gamma=1.0, threshold=0.00001)
    gridWorld.policyIteration()
    # gridWorld.printGridValue()
    # gridWorld.printOptimalPolicy()
    
if __name__ == '__main__':
    main()