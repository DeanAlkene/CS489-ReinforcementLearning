class MDP:
    def __init__(self, state, action, trans, reward, gamma, policy, threshold):
        self.state = state #S
        self.action = action #A
        self.trans = trans #P
        self.reward = reward #R
        self.gamma = gamma #gamma
        self.policy = policy #pi
        self.threshold = threshold #theta in IPE

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
    print("hello")

if __name__ == '__main__':
    main()