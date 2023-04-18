import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('models/policies/mobile_distance/logs/policy_replay_buffer_7500000_steps.pickle', 'rb') as rpb:
    replay_buffer = pickle.load(rpb)
with open('models/policies/mobile_distance/metadata.json', 'r') as m:
    metadata = json.load(m)

r = replay_buffer.reward
print(metadata)
r = r.reshape(len(r)//metadata['steps'], metadata['steps'])
plt.plot(np.arange(r.shape[0]), r.mean(axis=-1))
plt.show()