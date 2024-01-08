# AGRL: Autonomous Goal Representation Learning

Official code implementation of the `Autonomous Goal Representation Learning (AGRL)` pipeline (see our [paper](https://ieeexplore.ieee.org/abstract/document/10345879) for details).

## Main Code

The implementation of the code for developing goal-conditioned policies using latent state repersentation is located at `agrl/` folder.

### Reinforcement Learing Algorithms

The reinforcement learning algorithms are developed under the `agrl/rl/` folder. The implemented algorithms are `TD3`, `SAC` and `PPO`.

`SAC` and `PPO` algorithms are utilized from the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.

`TD3` implementation is an adapted implementation of the [author's pytorch version](https://github.com/sfujim/TD3).

### Evolutionary Strategies (Under development)

Evolutionary algorithms are implemented under `agrl/es/` folder.

### Controllers

In the folder `agrl/controllers/`, there are some basic implementation for controller suitable for differatial-drive. These are used to develop policies that output actions different than the robot's motors' commands.

### Robots

Robot implementations exist under the folder `agrl/robots/`. Until now, there are implementations of "point" (particle) agent, which only translates its body at x and y coordinates, and the quadruped robot, Anymal.

### Transformations

Using the concept of [torchvision's transforms](https://pytorch.org/vision/stable/transforms.html), an equivilent implementation is developed. A `transforms.Compose()` object can be created, in which differenet transformation operations can be added that are suitable for state datasets.

The `Compose` object is iteratable and every transform operation included can be called on the desired data. The `Compose` object can be based directly to the Data Loader, in order to perform the transformation automaticly.

```python
scaler = Scaler(_type='standard') # Transform Operation Object
shuffle = Shuffle(seed=42) # Transform Operation Object

transforms = Compose([
    shuffle,
    AngleToSinCos(angle_column=2),
    scaler.fit,
    scaler
])
dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=transforms)
```

### Variational Autoencoder

For creating a latent state representation, a Variational Autoencoder (VAE) is required, which maps the observations to lower dimension data. The VAE architecture, along with utils and the data loader, is at `agrl/vae/`.

### Enviroment

Under the `agrl/env/` folder, there are the implementations of different enviroments and helper classes to create custom enviroments to train the RL agent. The main component is the `BaseEnv` class, which includes the required operations to create the goal-conditioned policy. The envirometns are created using the OpenAI's Gym paradigm.

The methods in the `BaseEnv` class are:

* reset(self, *, seed=None, options=None): Resets the enviroment to its initial state.
* step(self, action): Steps into the enviroment and return the observation, reward and if the state is terminal.
* set_max_steps(self, max_steps): Sets the number of steps that define an episode.
* eval(self): Sets the enviroment into evalutation mode, so that the goal's latent representation is deterministic (not sampled from the latent space).
* train(self): Sets the enviroment back to training mode.
* _set_target(self): Sets the episode's target.
* _goal_conditioned_policy(self, target): Implements the goal-conditioned part of the policy. Initially a target is sampled from the exploration dataset and it is passed through the VAE's encoder. Later on, using the encoder's output (mean and sigma), a distribution is created from which a latent vector is sampled. This latent vector is passed through the decoder, which produces the reconstructed target given this vector. The reconstructed target becomes the episode's target.

There are also placeholders for methods that can be used to create the custom enviroments.
* _truncation_fn(self): Returns true if the state of the episode is terminal, but not goal is not achieved (e.g. stepped out of bounds).
* render(self): Graphics rendering implementation.
* close(self): Close rendering.
* _reward_fn(self, observation): Reward function implementation.
* _termination_fn(self, *args): Returns true if the episode is terminated successfully (goal achieved).
* _observations(self): Returns the agent's observations.
* _set_robot_state(self, state): Sets robots current state. Used in `reset()` method.
* _robot_act(self, action): Implementation of robot's act that is performed at each step.
* _state(self): Returns the enviroment's state.
* _reset_op(self): Append additional operations that should occure in the enviroment's reset.

## Datasets

Dataset for differenent experiments are located under `data/`. Evaluation data is located at `data/eval_data/`.

To generate a dataset from sampling a uniform distribution, use the `scripts/generate_uniform_data.py`

E.g:
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
**Evaluation Data Generation**: To generate evaluateion data, use the `scripts/generate_eval_data.py` script with the following arguments:

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

## Citing AGRL

If you use this code in a scientific publication, please use the following citation ([url](https://ieeexplore.ieee.org/abstract/document/10345879)):

```bibtex
@inproceedings{tsakoans2023effective,
        title={{Effective Skill Learning via Autonomous Goal Representation Learning}},
        author={Tsakonas, Constantinos and Chatzilygeroudis, Konstantinos},
        year={2023},
        booktitle={{The Fourteenth International Conference on Information, Intelligence, Systems and Applications (IISA 2023)}}
      }
```

## Acknowledgments

This work was supported by the [Hellenic Foundation for Research and Innovation](https://www.elidek.gr/en/homepage/) (H.F.R.I.) under the "3rd Call for H.F.R.I. Research Projects to support Post-Doctoral Researchers" (Project Acronym: NOSALRO, Project Number: 7541).

<p align="center">
<img src="https://www.elidek.gr/wp-content/themes/elidek/images/elidek_logo_en.png" alt="logo_elidek"/>
<p/>

<!-- <center>
<img src="https://nosalro.github.io/images/logo_elidek.png" alt="logo_elidek" width="50%"/>
</center> -->

This work was conducted within the [Computational Intelligence Lab](http://cilab.math.upatras.gr/) (CILab), Department of Mathematics, University of Patras, Greece.

<p align="center">
<img src="https://nosalro.github.io/images/logo_cilab.jpg" alt="logo_cilab" width="50%"/>
<img src="https://www.upatras.gr/wp-content/uploads/up_2017_logo_en.png" alt="logo_cilab" width="50%"/>
</p>

## License

[BSD 2-Clause "Simplified" License](https://opensource.org/license/bsd-2-clause/)
