#!/bin/bash
for exp_no in `seq 1 2`
do
    python experiments/alley_experiments_1d/mse_gc_latent_all_goals.py         --file-name models/policies/1d/alley_mobile_mse_${exp_no}	 --steps 120 --episodes 15000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 20 --checkpoint-episodes 200 --expl-noise 0.4 --batch-size 256 --scheduling-episode 1500
    python experiments/alley_experiments_1d/mse_gc_latent_all_goals_uniform.py --file-name models/policies/1d/alley_mobile_mse_uniform_${exp_no}  --steps 120 --episodes 15000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 20 --checkpoint-episodes 200 --expl-noise 0.4 --batch-size 256 --scheduling-episode 1500
    python experiments/alley_experiments_1d/edl_gc_latent_all_goals.py 	    --file-name models/policies/1d/alley_mobile_edl_${exp_no} 		 --steps 120 --episodes 15000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 20 --checkpoint-episodes 200 --expl-noise 0.4 --batch-size 256 --scheduling-episode 1500
    python experiments/alley_experiments_1d/edl_gc_latent_all_goals_uniform.py --file-name models/policies/1d/dalley_mobile_edl_uniform_${exp_no} --steps 120 --episodes 15000 --actor-lr 1e-3 --critic-lr 1e-3 --eval-freq 80000 --start-episode 20 --checkpoint-episodes 200 --expl-noise 0.4 --batch-size 256 --scheduling-episode 1500
done
