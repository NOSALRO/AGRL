import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def eval_logs():
    folder_path = sys.argv[1]
    logs = {}
    for root, dirs, files in os.walk(folder_path,topdown=True):
        for d in dirs:
            logs[d[:-2]] = []
        for d in dirs:
            _policy_eval_r = []
            for root, _, files in os.walk(f'{folder_path}/{d}/logs/'):
                _eval_files = []
                for f in files:
                    if f.split('_')[1] == 'eval':
                        _eval_files.append(int(f.split('_')[3]))
                _eval_files.sort(key=int)
                for f in _eval_files:
                    _policy_eval_r.append(np.mean(np.loadtxt(f'{folder_path}/{d}/logs/policy_eval_rewards_{f}_steps.dat'), axis=-1))
            logs[d[:-2]].append(np.array(_policy_eval_r))
        break
    legend = []
    for k, v in logs.items():
        plt.plot(np.median(v, axis=0))
        plt.fill_between(np.arange(len(v[0])), np.quantile(v,0.25, axis=0).astype(np.float32), np.quantile(v,0.75, axis=0).astype(np.float32), alpha=0.3)
        legend.append(k)
        legend.append(None)
    plt.legend(legend)


if __name__ == '__main__':
    plt.style.use('bmh')
    eval_logs()
    plt.show()