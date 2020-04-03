import math
import random

class GridWorld:
    def __init__(self, state, terminalState, gamma):
        self.state = state #|S|
        self.terminalState = terminalState
        self.gamma = gamma #gamma

        self.value = []
        self.policy = [] #pi[s][a]
        self.gridSize = int(math.sqrt(self.state))
        self.action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3} #A
        self.trans = [[0 for j in range(len(self.action))] for i in range(self.state)] #s' = trans[s][a] 
        self.prob = [[[0.0 for k in range(self.state)] for j in range(len(self.action))] for i in range(self.state)] #P[s][a][s']
        self.reward = [[-1.0 for j in range(len(self.action))] for i in range(self.state)] #E[R[s][a]]
        self.map = [[(j + i * self.gridSize) for j in range(self.gridSize)] for i in range(self.gridSize)] #for calculating trans[s][a] and filling in P[s][a][s']
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

    def MC(self):
        pass

    def TD0(self):
        pass

    def printInfo(self):
        print("Gridworld with %d states:"%(self.state))
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                print("%d\t"%(self.map[i][j]), end='')
            print()
        print("Terminal States: ", end='')
        for ts in self.terminalState:
            print("%d "%(ts), end='')
        print("\n")

    def printGridValue(self):
        for i in range(self.gridSize):
            for j in range(self.gridSize):
                curState = j + i * self.gridSize
                print("%0.2f\t"%(self.value[curState]), end='')
            print()

    def validationTest(self):
        for i in range(self.state):
            for a in self.action.keys():
                for j in range(self.state):
                    if not math.fabs(self.prob[i][self.action[a]][j]) <= 1e-6:
                        print("state %d in action %s to state %d has prob %f"%(i, a, j, self.prob[i][self.action[a]][j]))
            print()

def main():
    gridWorld = GridWorld(state=36, terminalState=[1, 35], gamma=1.0)
    gridWorld.printInfo()
    print("\nMC:")
    gridWorld.MC()
    gridWorld.printGridValue()
    print("\nTD(0):")
    gridWorld.TD0()
    gridWorld.printGridValue()
    
if __name__ == '__main__':
    main()