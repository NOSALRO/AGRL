#!/bin/bash
for exp_no in `seq 11 20`
do
    echo "Running alley_experiments_1d/mse_gc_latent_all_goals.py: ${exp_no}"
    python experiments/alley_experiments_1d/mse_gc_latent_all_goals.py         --file-name models/policies/1d_gnll_more_episodes/alley_mobile_mse_gnll_ge_${exp_no}	 --steps 130 --episodes 4000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 10 --checkpoint-episodes 40 --expl-noise 0.2 --batch-size 256 --scheduling-episode 0

    echo "Running alley_experiments_1d/mse_gc_latent_all_goals_uniform.py: ${exp_no}"
   python experiments/alley_experiments_1d/mse_gc_latent_all_goals_uniform.py --file-name models/policies/1d_gnll_more_episodes/alley_mobile_mse_gnll_uniform_${exp_no}  --steps 130 --episodes 4000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 10 --checkpoint-episodes 40 --expl-noise 0.2 --batch-size 256 --scheduling-episode 0
done
