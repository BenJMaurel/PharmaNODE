#!/bin/bash
export OMP_NUM_THREADS=1
nohup python3 run_models.py --niters 40000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed 1 --experiment confound_km09 --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 --decoder-hidden 50 > logs/exp2_film_km09_decoder.log 2>&1 &
echo "Started km09"
nohup python3 run_models.py --niters 40000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed 1 --experiment confound_km00 --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 --decoder-hidden 50 > logs/exp2_film_km00_decoder.log 2>&1 &
echo "Started km00"
