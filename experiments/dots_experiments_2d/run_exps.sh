#!/bin/bash
for exp_no in `seq 6 10`
do
    echo "Running dots_experiments_2d/edl_gc_latent_all_goals.py"
    python experiments/dots_experiments_2d/edl_gc_latent_all_goals.py 	        --file-name models/policies/2d/dots_mobile_edl_${exp_no}         --steps 200 --episodes 24000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 100 --expl-noise 0.4 --batch-size 256 --scheduling-episode 3000

    echo "Running dots_experiments_2d/mse_gc_latent_all_goals.py"
#     python experiments/dots_experiments_2d/mse_gc_latent_all_goals.py 	        --file-name models/policies/2d/dots_mobile_mse_${exp_no}         --steps 200 --episodes 24000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 100 --expl-noise 0.4 --batch-size 256 --scheduling-episode 4000

    # echo "Running dots_experiments_2d/mse_gc_latent_all_goals_uniform.py"
    # python experiments/dots_experiments_2d/mse_gc_latent_all_goals_uniform.py 	--file-name models/policies/2d/dots_mobile_mse_uniform_${exp_no} --steps 200 --episodes 20000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 1000 --expl-noise 0.4 --batch-size 256 --scheduling-episode 4000

    # echo "Running dots_experiments_2d/edl_gc_latent_all_goals_uniform.py"
    # python experiments/dots_experiments_2d/edl_gc_latent_all_goals_uniform.py 	--file-name models/policies/2d/dots_mobile_edl_uniform_${exp_no} --steps 200 --episodes 40000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 60 --checkpoint-episodes 250 --expl-noise 0.4 --batch-size 256 --scheduling-episode 5000
done
