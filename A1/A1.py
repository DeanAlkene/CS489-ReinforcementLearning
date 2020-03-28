import math

class GridWorld:
    def __init__(self, state, terminalState, gamma, threshold):
        self.state = state #S
        self.terminalState = terminalState
        self.gamma = gamma #gamma
        self.threshold = threshold #theta in IPE

        self.action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3} #A
        self.trans = [[0 for j in range(len(action))] for i in range(state)] #s' = trans[s][a] 
        self.prob = [[[0 for k in range(state)] for j in range(len(action))] for i in range(state)] #P[s][a][s']
        self.reward = [[-1 for j in range(len(action))] for i in range(state)] #E[R[s][a]]
        self.policy = [[0.25 for j in range(len(action))] for i in range(state)] #pi
        self.map = [[(i+1)*(j+1)-1 for j in range(int(math.sqrt(state)))] for i in range(int(math.sqrt(state)))]
        self.__calcParam()

    def __calcParam(self):
        

    def evaluation(self):
        pass

    def policyIteration(self):
        pass

    def valueIteration(self):
        pass

    def printGridValue(self):
        pass

    def printOptimalPolicy(self):
        pass

def main():
    gridWorld = GridWorld(state=36, terminalState=[1,35], gamma=1, threshold=0.0001)


if __name__ == '__main__':
    main()