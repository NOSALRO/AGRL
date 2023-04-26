#!/bin/bash
for exp_no in `seq 1 5`
do
    echo "Running maze_experiments_1d/mse_gc_latent_all_goals.py: ${exp_no}"
    python experiments/maze_experiments_1d/mse_gc_latent_all_goals.py         --file-name models/policies/1d/maze_mobile_mse_ge_${exp_no}	 --steps 130 --episodes 3000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 30 --checkpoint-episodes 30 --expl-noise 0.2 --batch-size 256 --scheduling-episode 1500

    echo "Running maze_experiments_1d/mse_gc_latent_all_goals_uniform.py: ${exp_no}"
    python experiments/maze_experiments_1d/mse_gc_latent_all_goals_uniform.py --file-name models/policies/1d/maze_mobile_mse_uniform_${exp_no}  --steps 130 --episodes 3000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 30 --checkpoint-episodes 30 --expl-noise 0.2 --batch-size 256 --scheduling-episode 1500
done
