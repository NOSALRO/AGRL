import argparse

def cli(STEPS, EPISODES):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="choose RL algorithm (PPO/SAC).")
    parser.add_argument("-m", "--mode", default='train', help="choose between train, continue and eval mode.")
    parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics during training.")
    # parser.add_argument("--map", default="three_wall", help="choose map. Default: three_wall")
    parser.add_argument("--file-name", default=None, help="file name to save model. If eval mode is on, then name of the model to load.")
    parser.add_argument("--steps", default=STEPS, help="number of env steps. Default: 1e+03")
    parser.add_argument("--episodes", default=EPISODES, help="number of episodes. Default: 1e+03")
    args = parser.parse_args()
    algorithm = args.algorithm
    mode = args.mode
    graphics = args.graphics
    # map_name = args.map
    file_name = args.file_name if args.file_name else algorithm.lower()
    file_name = '.'.join(file_name.split('.')[:-1]) if file_name.split('.')[-1] == 'zip' else file_name
    STEPS = int(args.steps)
    EPISODES = int(args.episodes)
    return algorithm, mode, graphics, file_name, STEPS, EPISODES