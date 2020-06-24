import argparse
import A3C
import DDPG
import DDPG_noise
import DQN

legal_env = [['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'],
             ['Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Ant-v2']]

legal_method = [['DQN'], ['A3C', 'DDPG', 'DDPG_noise']]

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name')
    parser.add_argument('--method')

    args = parser.parse_args()
    if args.env_name not in legal_env[0] and args.env_name not in legal_env[1]:
        print('Environment Not Supported')
        return
    if args.method not in legal_method[0] and args.method not in legal_method[1]:
        print('Algorithm Not Supported')
        return
    if args.env_name in legal_env[0] and args.method in legal_method[0]:
        DQN.runDQN(args.env_name)
    elif args.env_name in legal_env[1] and args.method in legal_method[1]:
        if args.method == 'A3C':
            A3C.runA3C(args.env_name)
        elif args.method == 'DDPG':
            DDPG.runDDPG(args.env_name)
        elif args.method == 'DDPG_noise':
            DDPG_noise.runDDPG(args.env_name)
        else:
            pass
    else:
        print('Not suitable Environment & Algorithm')
        return

if __name__ == '__main__':
    run()