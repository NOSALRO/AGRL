#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Constantinos Tsakonas
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Constantinos Tsakonas
#|              Konstantinos Chatzilygeroudis
#|    email:    tsakonas.constantinos@gmail.com
#|              costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of AGRL.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

def eval_logs():
    folder_path = sys.argv[1]
    logs = {}
    for root, dirs, files in os.walk(folder_path,topdown=True):
        for d in dirs:
            logs['_'.join(d.split('_')[:-1])] = []
        for d in dirs:
            _policy_eval_r = []
            for root, _, files in os.walk(f'{folder_path}/{d}/logs/'):
                _eval_files = []
                for f in files:
                    if f.split('_')[1] == 'final':
                        _eval_files.append(int(f.split('_')[3]))
                _eval_files.sort(key=int)
                for f in _eval_files:
                    _policy_eval_r.append(np.mean(np.loadtxt(f'{folder_path}/{d}/logs/policy_eval_rewards_{f}_steps.dat'), axis=-1))
                print(f'{folder_path}/{d}/logs/policy_eval_rewards_{f}_steps.dat')
                print(np.array(_policy_eval_r).shape)
            logs['_'.join(d.split('_')[:-1])].append(np.array(_policy_eval_r))
        break
    legend = []
    for k, v in logs.items():
        plt.plot(np.mean(v, axis=0))
        plt.fill_between(np.arange(len(v[0])), np.mean(v, axis=0)-np.std(v, axis=0).astype(np.float32), np.mean(v, axis=0) + np.std(v, axis=0).astype(np.float32), alpha=0.3, label='_nolegend_')
        legend.append(k)
    #data1, data2 = logs.values()
    #data1 = np.array(data1)
    #data2 = np.array(data2)
    #z, p = scipy.stats.mannwhitneyu(data1[:,-1], data2[:,-1])
    #p_value = p * 2
    #print(stars(p_value))
    return legend, logs


if __name__ == '__main__':
    plt.style.use('bmh')
    legend_map = {
        "iiwa_ge" : "Iiwa Go-Explore",
        "iiwa_uniform": "Iiwa Uniform",
        "iiwa_uniform_baseline": "Iiwa Uniform Baseline",
        "dots_mobile_edl": "Mobile Obstacles Logprob Reward",
        "dots_mobile_mse": "Mobile Obstacles MSE Reward",
        "alley_mobile_mse_gnll_ge": "Mobile Line Go-Explore",
        "alley_mobile_mse_gnll_uniform": "Mobile Line Uniform",
        "dots_mobile_mse_discrete": "Mobile Obstacles Skill-Conditioned",
        "dots_mobile_mse_continuous": "Mobile Obstacles Goal-Conditioned",
    }
    legend, logs = eval_logs()

    for idx,_ in enumerate(legend):
        legend[idx] = legend_map[legend[idx]]
    plt.tight_layout()
    plt.legend(legend, bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=5)

    # plt.savefig(sys.argv[2], bbox_inches='tight')
    for k, v in logs.items():
        np.savetxt(f'figure_data/{k}.dat', v)
