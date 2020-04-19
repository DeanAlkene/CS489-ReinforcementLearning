# CS489 Assignment 3 Report

517030910214 Hongzhou Liu

## 0. Introduction

In this assignment, we are required to implement SARSA and Q-Learning in CliffWalking environment. CliffWalking is a variant of GridWorld. In CliffWalking, the start state is specified and there's a cliff lying between start state and goal state. Once our agent moves onto the cliff, it slips directly to the start state and receives a reward of -100. In other case, a move will incur a reward of -1.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/0.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.1 CliffWalking</div>
</center>


Environment:

- Ubuntu 18.04 LTS
- Python 3.7.7

```python
for cliffState in range(self.startState + 1, self.goalState):
            edgeState = self.trans[cliffState][self.action['n']]
            self.trans[edgeState][self.action['s']] = self.startState
            self.reward[edgeState][self.action['s']] = -100.0
        self.trans[self.startState][self.action['e']] = self.startState
        self.reward[self.startState][self.action['e']] = -100.0
```

## 1. SARSA

### 1.1 Implementation

### 1.2 Result

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/1.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.2 SARSA, epsilon=0.00001</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/2.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.3 SARSA, epsilon=0.0001</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/3.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.4 SARSA, epsilon=0.001</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/4.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.5 SARSA, epsilon=0.1</div>
</center>

## 2. Q-Learning

### 2.1 Implementation

### 2.2 Result

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/5.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.6 Q-Learning, epsilon=0.001</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/6.png"
    width=500>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Fig.7 Q-Learning, epsilon=0.1</div>
</center>

## 3. Summary