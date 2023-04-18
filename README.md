# Goal-Conditioned Policies using Latent State Representations

## Main Code

The implementation of the code for developing goal-conditioned policies using latent state repersentation is located at `nosalro/` folder.

### Reinforcement Learing Algorithms

The reinforcement learning algorithms are developed under the `nosalro/rl/` folder. The implemented algorithms are `TD3`, `SAC` and `PPO`.

`SAC` and `PPO` algorithms are utilized from the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.

`TD3` implementation originates from this [github repository](https://github.com/sfujim/TD3).

### Evolutionary Strategies (Under development)

Evolutionary algorithms are implemented under `nosalro/es/` folder.

### Controllers

In the folder `nosalro/controllers/`, there are some basic implementation for controller suitable for differatial-drive. These are used to develop policies that output actions different than the robot's motors' commands.

### Robots

Robot implementations exist under the folder `nosalro/robots/`. Until now, there are implementations of "point" (particle) agent, which only translates its body at x and y coordinates, and the quadruped robot, Anymal.

### Transformations

Using the concept of [torchvision's transforms](https://pytorch.org/vision/stable/transforms.html), an equivilent implementation is developed. A `transforms.Compose()` object can be created, in which differenet transformation operations can be added that are suitable for state datasets.

The `Compose` object is iteratable and every transform operation included can be called on the desired data. The `Compose` object can be based directly to the Data Loader, in order to perform the transformation automaticly.

```python
scaler = Scaler(_type='standard') # Transform Op Objec
shuffle = Shuffle(seed=42) # Transform Op

transforms = Compose([
    shuffle,
    AngleToSinCos(angle_column=2),
    scaler.fit,
    scaler
])
dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=transforms)
```

### Variational Autoencoder

## Datasets

Dataset for differenent experiments are located under `data/`. Evaluation data is located at `data/eval_data/`.

To generate a dataset from sampling a uniform distribution, use the `scripts/generate_uniform_data.py`

E.g.
```
python scripts/generate_uniform_data.py --N 540 --file-name data/uniform_alley.dat
```

## Training Policies

Under the folder experiments, there is a subset of experiments configurations based on goal-contitioned policies for a mobile robot implementing differenatial drive. The possible configurations for the experiments are

```
-h, --help                                    show this help message and exit
--file-name FILE_NAME                         File name to save model. If eval mode is on, then name of the model to load.
--steps STEPS                                 Number of env steps. Default: 1e+03
--episodes EPISODES                           Number of episodes. Default: 1e+05
--start-episode START_EPISODE                 Number of episodes initial random policy is used
--eval-freq EVAL_FREQ                         How often (episodes) the policy is evaluated
--actor-lr ACTOR_LR                           Actor learning rate
--critic-lr CRITIC_LR                         Critic learning rate
--expl-noise EXPL_NOISE                       Std of Gaussian exploration noise
--batch-size BATCH_SIZE                       Batch size for both actor and critic
--discount DISCOUNT                           Discount factor
--tau TAU                                     Target network update rate
--policy-noise POLICY_NOISE                   Noise added to target policy during critic update
--noise-clip NOISE_CLIP                       Range to clip target policy noise
--policy-freq POLICY_FREQ                     Frequency of delayed policy updates
--update-steps UPDATE_STEPS                   Number of update steps for on-policy algorithms.
--epochs EPOCHS                               Number of epochs for policy update.
--checkpoint-episodes CHECKPOINT_EPISODES     Nubmer of episodes a checkpoint is stored.
--seed SEED                                   Enviroment seed.
-g, --graphics                                Enable graphics during training.
```

Example of running an experiment:

```
python experiments/alley_experiments/edl_gc_latent_all_goals.py --file-name models/policies/alley_mobile_edl \
                                                                --steps 100 \
                                                                --episodes 50000 \
                                                                --actor-lr 1e-3 \
                                                                --critic-lr 1e-3 \
                                                                --eval-freq 10000 \
                                                                --start-episode 1000 \
                                                                --checkpoint-episodes 5000 \
                                                                --expl-noise 0.5 \
                                                                --batch-size 512
```


## Policy evaluation

Policies are stored under the foler path `models/policies/`. To evaluate a policy execute the  `eval_policy.py` script.

```shell
python scripts/eval_policy.py <policy_to_evaluate_path> [<evaluation_data_path>]
```

E.g:

```shell
python scripts/eval_policy.py models/policies/alley_mobile_distance_uniform/ data/eval_data/alley.dat
```
### Evaluation Data Generation: To generate evaluateion data, use the `scripts/generate_eval_data.py` script with the following arguments:

```
-h, --help              show this help message and exit
--N N                   Number of generated points.
--map MAP               map file path. Choose between train, continue and eval mode.
--file-name FILE_NAME   file name to save the data.
-g, --graphics          enable graphics to preview the targets.
```
E.g: Generate evaluation data:
```
python scripts/generate_eval_data.py --file-name data/eval_data/alley.dat --map worlds/alley.pbm --N 500
```