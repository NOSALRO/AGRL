import argparse
import torch
import numpy as np
import pyfastsim as fastsim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type= int, help="Number of generated points.")
    parser.add_argument("--file-name", default=None, help="file name to save the data.")
    args = parser.parse_args()
    goals = np.random.uniform(low=0, high=600, size=(args.N, 2))
    if args.file_name is not None:
        np.savetxt(args.file_name, goals)
