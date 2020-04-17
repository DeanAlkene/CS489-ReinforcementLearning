import math
import random

class CliffWalking:
    def __init__(self, width, length, startState, goalState, gamma):
        self.width = width
        self.length = length
        self.state = width * length #|S|
        self.startState = startState
        self.goalState = goalState
        self.gamma = gamma #gamma

        self.value = []
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

        for cliffState in range(self.startState + 1, self.goalState):
            edgeState = self.trans[cliffState][self.action['n']]
            self.trans[edgeState][self.action['s']] = self.startState
            self.reward[edgeState][self.action['s']] = -100.0
        self.trans[self.startState][self.action['e']] = self.startState
        self.reward[self.startState][self.action['e']] = -100.0
        self.trans[self.goalState][self.action['w']] = self.startState
        self.reward[self.goalState][self.action['w']] = -100.0

        tmp = self.trans[self.goalState][self.action['n']]
        self.trans[tmp][self.action['s']] = self.startState + 1
        tmp = self.trans[self.goalState][self.action['e']]
        self.trans[tmp][self.action['w']] = self.startState + 1
        tmp = self.trans[self.goalState][self.action['s']]
        self.trans[tmp][self.action['n']] = self.startState + 1
        tmp = self.trans[self.goalState][self.action['w']]
        self.trans[tmp][self.action['e']] = self.startState + 1

        for i in range(self.startState + 1, self.goalState):
            self.map[i//self.length][i%self.length] = '--'
        self.map[self.width-1][self.length-1] = self.startState + 1

        self.state -= self.goalState - self.startState - 1
        self.goalState = self.startState + 1
        for a in self.action.keys():
            self.reward[self.goalState][self.action[a]] = 0.0
    # def __episodeGenerator(self):
    #     episode = []
    #     episode.append(random.randint(0, 35))
    #     curState = episode[0]
    #     while curState not in self.terminalState:
    #         curAction = random.randint(0, 3)
    #         curReward = self.reward[curState][curAction]
    #         curState = self.trans[curState][curAction]
    #         episode.extend([curAction, curReward, curState])
    #     return episode

    def printInfo(self):
        print("Cliff Walking with %d states:"%(self.state))
        for i in range(self.width):
            for j in range(self.length):
                print("%s\t"%(str(self.map[i][j])), end='')
            print()
        print("Start State: %d"%(self.startState))
        print("Goal State: %d"%(self.goalState))

    # def printGridValue(self):
    #     for i in range(self.gridSize):
    #         for j in range(self.gridSize):
    #             curState = j + i * self.gridSize
    #             print("%0.2f\t"%(self.value[curState]), end='')
    #         print()
    
    def validationCheck(self):
        for s in range(self.state):
            print("State: %d"%(s))
            for a in ['n', 'e', 's', 'w']:
                print("%s: %d, %0.1f"%(a, self.trans[s][self.action[a]], self.reward[s][self.action[a]]))

def main():
    cliffWalking = CliffWalking(width=4, length=12, startState=36, goalState=47, gamma=1.0)
    cliffWalking.printInfo()
    cliffWalking.validationCheck()
    
if __name__ == '__main__':
    main()
