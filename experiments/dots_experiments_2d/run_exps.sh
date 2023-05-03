#!/bin/bash
for exp_no in `seq 1 20`
do
    # echo "Running dots_experiments_2d/edl_gc_latent_all_goals.py"
    # python experiments/dots_experiments_2d/edl_gc_latent_all_goals.py 	        --file-name models/policies/2d/dots_mobile_edl_${exp_no}         --steps 200 --episodes 24000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 100 --expl-noise 0.4 --batch-size 256 --scheduling-episode 3000

    # echo "Running dots_experiments_2d/mse_gc_latent_all_goals.py"
    # python experiments/dots_experiments_2d/mse_gc_latent_all_goals.py 	        --file-name models/policies/2d/dots_mobile_mse_${exp_no}         --steps 200 --episodes 24000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 100 --expl-noise 0.4 --batch-size 256 --scheduling-episode 4000

    echo "Running dots_experiments_2d/mse_gc_latent_all_goals_discrete.py"
    python experiments/dots_experiments_2d/mse_gc_latent_discrete_mse.py 	        --file-name models/policies/2d_discrete/dots_mobile_mse_discrete_${exp_no} --steps 200 --episodes 24000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 100 --expl-noise 0.4 --batch-size 256 --scheduling-episode 3000
done
