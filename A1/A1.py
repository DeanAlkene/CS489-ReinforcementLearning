class MDP:
    def __init__(self, state, action, trans, reward, gamma, policy, threshold):
        self.state = state #S
        self.terminalState = []
        self.action = action #A
        self.trans = trans #P
        self.reward = reward #R
        self.gamma = gamma #gamma
        self.policy = policy #pi
        self.threshold = threshold #theta in IPE

    def setTerminal(self, termList):
        self.terminalState = termList

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
    action = {'n' : 0, 'e' : 1, 's' : 2, 'w' : 3}
    policy = []
    gridWorld = MDP()
    gridWorld.setTerminal([1, 35])


if __name__ == '__main__':
    main()