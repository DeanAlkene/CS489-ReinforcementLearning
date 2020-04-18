import math
import random
import copy
import numpy as np

class CliffWalking:
    def __init__(self, width, length, startState, goalState, gamma):
        self.width = width
        self.length = length
        self.state = width * length #|S|
        self.startState = startState
        self.goalState = goalState
        self.gamma = gamma #gamma

        self.QValue = []
        self.action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3} #A
        self.policy = [] #pi[s][a]
        self.trans = [[0 for j in range(len(self.action))] for i in range(self.state)] #s' = trans[s][a] 
        self.reward = [[-1.0 for j in range(len(self.action))] for i in range(self.state)] #E[R[s][a]]
        self.map = [[(j + i * self.length) for j in range(self.length)] for i in range(self.width)] #for calculating trans[s][a] and filling in P[s][a][s']
        self.__calcParam()

    def __calcParam(self):
        curState = 0;
        for i in range(self.width): #row_index
            for j in range(self.length): #col_index
                if i == 0:
                    self.trans[curState][self.action['n']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['n']] = self.map[i-1][j]
                if i == self.width - 1:
                    self.trans[curState][self.action['s']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['s']] = self.map[i+1][j]
                if j == 0:
                    self.trans[curState][self.action['w']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['w']] = self.map[i][j-1]
                if j == self.length - 1:
                    self.trans[curState][self.action['e']] = self.map[i][j]
                else:
                    self.trans[curState][self.action['e']] = self.map[i][j+1]
                curState += 1
        for i in range(self.startState + 1, self.goalState):
            self.map[i//self.length][i%self.length] = '--'
        self.map[self.width - 1][self.length - 1] = self.startState + 1
        
        for cliffState in range(self.startState + 1, self.goalState):
            edgeState = self.trans[cliffState][self.action['n']]
            self.trans[edgeState][self.action['s']] = self.startState
            self.reward[edgeState][self.action['s']] = -100.0
        self.trans[self.startState][self.action['e']] = self.startState
        self.reward[self.startState][self.action['e']] = -100.0

        originGoal = self.goalState
        self.state -= self.goalState - self.startState - 1
        self.goalState = self.startState + 1

        for a in self.action.keys():
            if self.trans[originGoal][self.action[a]] != originGoal:
                self.trans[self.goalState][self.action[a]] = self.trans[originGoal][self.action[a]]
            else:
                self.trans[self.goalState][self.action[a]] = self.goalState
        self.trans[self.goalState][self.action['w']] = self.startState
            
        for a in self.action.keys():
            if self.trans[self.goalState][self.action[a]] != self.goalState and self.trans[self.goalState][self.action[a]] != self.startState:
                tmp = self.trans[self.goalState][self.action[a]]
                self.trans[tmp][(self.action[a] + 2) % 4] = self.goalState
        
        for a in self.action.keys():
            self.reward[self.goalState][self.action[a]] = 0.0

    def __actionGenerator(self, s, epsilon):
        prob = random.random()
        accProb = [sum(self.policy[s][:i]) for i in range(len(self.action) + 1)]
        for i in range(len(self.action)):
            if prob >= accProb[i] and prob < accProb[i + 1]:
                a = i
        return a

    def __updatePolicy(self, s, epsilon):
        optimalAction = np.argmax(self.QValue[s])
        for a in self.action.values():
            if a != optimalAction:
                self.policy[s][a] = epsilon / len(self.action)
            else:
                self.policy[s][a] = 1 - epsilon + epsilon / len(self.action)
                
    def SARSA(self, alpha, epsilon, iterTimes):
        #Initialize
        self.QValue = [[random.random() for j in range(len(self.action))] for i in range(self.state)]
        self.policy = [[0.0 for j in range(len(self.action))] for i in range(self.state)]
        for a in self.action.values():
            self.QValue[self.goalState][a] = 0.0
        for s in range(self.state):
            self.__updatePolicy(s, epsilon)

        #Iteration
        for _ in range(iterTimes):
            curState = self.startState
            curAction = self.__actionGenerator(curState, epsilon)
            while curState != self.goalState:        
                nextState = self.trans[curState][curAction]
                curReward = self.reward[curState][curAction]
                nextAction = self.__actionGenerator(nextState, epsilon)
                self.QValue[curState][curAction] = self.QValue[curState][curAction] + alpha * (curReward + self.gamma * self.QValue[nextState][nextAction] - self.QValue[curState][curAction])
                self.__updatePolicy(curState, epsilon)
                curState = nextState
                curAction = nextAction

    def Q_Learning(self, alpha, epsilon, iterTimes):
        #Initialize
        self.QValue = [[random.random() for j in range(len(self.action))] for i in range(self.state)]
        self.policy = [[0.0 for j in range(len(self.action))] for i in range(self.state)]
        for a in self.action.values():
            self.QValue[self.goalState][a] = 0.0
        for s in range(self.state):
            self.__updatePolicy(s, epsilon)

        #Iteration
        for _ in range(iterTimes):
            curState = self.startState
            while curState != self.goalState:
                curAction = self.__actionGenerator(curState, epsilon)
                nextState = self.trans[curState][curAction]
                curReward = self.reward[curState][curAction]
                self.QValue[curState][curAction] = self.QValue[curState][curAction] + alpha * (curReward + self.gamma * max(self.QValue[nextState]) - self.QValue[curState][curAction])
                self.__updatePolicy(curState, epsilon)
                curState = nextState
        
    def printOptimalPolicy(self):
        reverseTable = ['n', 'e', 's', 'w']
        opt = []
        curState = self.startState
        res = copy.deepcopy(self.map)
        while curState != self.goalState:
            opt.append(curState)
            opt.append(np.argmax(self.QValue[curState]))
            curState = self.trans[curState][np.argmax(self.QValue[curState])]
        for i in range(self.width):
            for j in range(self.length):
                if str(res[i][j]) != '--':
                    res[i][j] = 0
                else:
                    res[i][j] = '-'
        for i in range(0, len(opt), 2):
            curState = opt[i]
            res[curState // self.length][curState % self.length] = reverseTable[opt[i + 1]]
        res[self.width - 1][self.length - 1] = 'G'
        for i in range(self.width):
            for j in range(self.length):
                print("%s\t"%(str(res[i][j])), end='')
            print()

    def printInfo(self):
        print("Cliff Walking with %d states:"%(self.state))
        for i in range(self.width):
            for j in range(self.length):
                print("%s\t"%(str(self.map[i][j])), end='')
            print()
        print("Start State: %d"%(self.startState))
        print("Goal State: %d"%(self.goalState))

    def printGridQValue(self):
        for s in range(self.state):
            print("Q[%d]: " % (s), end='')
            print(self.QValue[s])
    
    def validationCheck(self):
        for s in range(self.state):
            print("State: %d"%(s))
            for a in ['n', 'e', 's', 'w']:
                print("%s: %d, %0.1f" % (a, self.trans[s][self.action[a]], self.reward[s][self.action[a]]))

def main():
    cliffWalking = CliffWalking(width=4, length=12, startState=36, goalState=47, gamma=1.0)
    cliffWalking.printInfo()
    for e in [0.00001, 0.0001, 0.001, 0.1]:
        cliffWalking.SARSA(alpha=0.2, epsilon=e, iterTimes=10000)
        print("SARSA, alpha=0.2, epsilon=%f:"%(e))
        cliffWalking.printOptimalPolicy()
        print()
        cliffWalking.Q_Learning(alpha=0.2, epsilon=e, iterTimes=10000)
        print("Q-Learning, alpha=0.2, epsilon=%f:"%(e))
        cliffWalking.printOptimalPolicy()
        print()
    
if __name__ == '__main__':
    main()
