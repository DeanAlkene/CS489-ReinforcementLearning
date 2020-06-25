# CS489 Reinforcement Learning Final Project
This folder contains source codes, trained models, report and demo videos.
## Training
You can train Dueling Double DQN, A3C, DDPG and DDPG with Parameter Space Noise by running the following command:
```bash
python run.py --env_name env --method method
```

You should specify the environment name and algorithm you want to use in the command. The environment "VideoPinball-ramNoFrameskip-v4" is currently not supported.

## Testing

You can test the result by loading the model stored in `./model` by modifying `test.py` and then run

```bash
python test.py
```

## Files

The files in this project is organized in the following way:

- Source code:
    - `./A3C.py`: A3C
    - `./DDPG.py`: DDPG
    - `./DDPG_noise.py`: DDPG with Parameter Space Noise
    - `./DQN,py`: DQN, Double DQN, Dueling DQN and Dueling Double DQN
    - `./run.py`: The code to train
    - `./test.py`: The code to test
- Saved models: 
    - `./model`: Saved models of experiments mentioned in the report
- Demo videos:
    - `./demo`: Some typical testing results
- Report:
    - `./Report`: PDF file of report