#!/bin/bash
for exp_no in `seq 1 10`
do
    echo "Running iiwa_experiments/iiwa_ge.py: ${exp_no}"
    # python experiments/iiwa_experiments/iiwa_ge.py       --file-name models/policies/iiwa/iiwa_ge_${exp_no}     --steps 200 --episodes 2000 --actor-lr 1e-3 --critic-lr 1e-3 --start-episode 10 --checkpoint-episodes 40 --expl-noise 0.2 --batch-size 256 --scheduling-episode 0

    echo "Running iiwa_experiments/iiwa_uniform.py: ${exp_no}"
    # python experiments/iiwa_experiments/iiwa_uniform.py  --file-name models/policies/iiwa/iiwa_uniform_${exp_no} --steps 200 --episodes 2000 --actor-lr 1e-3 --critic-lr 1e-3 --start-episode 10 --checkpoint-episodes 40 --expl-noise 0.2 --batch-size 256 --scheduling-episode 0

    echo "Running iiwa_experiments/iiwa_uniform_baseline.py: ${exp_no}"
    python experiments/iiwa_experiments/iiwa_uniform_baseline.py       --file-name models/policies/iiwa/iiwa_uniform_baseline_${exp_no}     --steps 200 --episodes 2000 --actor-lr 1e-3 --critic-lr 1e-3 --start-episode 10 --checkpoint-episodes 40 --expl-noise 0.2 --batch-size 256 --scheduling-episode 0
done
