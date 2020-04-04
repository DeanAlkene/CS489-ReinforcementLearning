import math
import random

class GridWorld:
    def __init__(self, state, terminalState, gamma):
        self.state = state #|S|
        self.terminalState = terminalState
        self.gamma = gamma #gamma

        self.value = []
        self.gridSize = int(math.sqrt(self.state))
        self.action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3} #A
        self.policy = [[0.25 for j in range(len(self.action))] for i in range(state)] #pi[s][a]
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
        
    def __episodeGenerator(self):
        episode = []
        episode.append(random.randint(0, 35))
        curState = episode[0]
        while curState not in self.terminalState:
            curAction = random.random()
            if curAction < 0.25:
                curAction = 0
            elif curAction >= 0.25 and curAction < 0.5:
                curAction = 1
            elif curAction >= 0.5 and curAction < 0.75:
                curAction = 2
            elif curAction >= 0.75:
                curAction = 3
            curReward = self.reward[curState][curAction]
            curState = self.trans[curState][curAction]
            episode.extend([curAction, curReward, curState])
        return episode

    def firstVisitMC(self, iterTime):
        #Initialize
        self.value = [0.0 for i in range(self.state)]
        counter = [0 for i in range(self.state)]
        returns = [0.0 for i in range(self.state)]
        visited = [False for i in range(self.state)]

        #Sample & Evaluate
        for i in range(iterTime):
            episode = self.__episodeGenerator()
            for j in range(0, len(episode), 3):
                curState = episode[j]
                if not visited[curState]:
                    visited[curState] = True
                    counter[curState] += 1
                    tmp = 0.0
                    decay = 1.0
                    for k in range(j + 2, len(episode), 3):
                        tmp += decay * episode[k]
                        decay *= self.gamma
                    returns[curState] += tmp
                    self.value[curState] = returns[curState] / counter[curState]

    def everyVisitMC(self, iterTime):
        #Initialize
        self.value = [0.0 for i in range(self.state)]
        counter = [0 for i in range(self.state)]
        returns = [0.0 for i in range(self.state)]

        #Sample & Evaluate
        for i in range(iterTime):
            episode = self.__episodeGenerator()
            for j in range(0, len(episode), 3):
                curState = episode[j]
                counter[curState] += 1
                tmp = 0.0
                decay = 1.0
                for k in range(j + 2, len(episode), 3):
                    tmp += decay * episode[k]
                    decay *= self.gamma
                returns[curState] += tmp
                self.value[curState] = returns[curState] / counter[curState]

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

def main():
    gridWorld = GridWorld(state=36, terminalState=[1, 35], gamma=1.0)
    gridWorld.printInfo()
    print("\nFirst Visit MC:")
    gridWorld.firstVisitMC(iterTime=10000)
    gridWorld.printGridValue()
    print("\nEvery Visit MC:")
    gridWorld.everyVisitMC(iterTime=1000)
    gridWorld.printGridValue()
    # print("\nTD(0):")
    # gridWorld.TD0()
    # gridWorld.printGridValue()
    
if __name__ == '__main__':
    main()