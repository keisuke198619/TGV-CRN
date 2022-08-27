#!/bin/bash

#### model training 

# carla # --small
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model GVTBCRN # full
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model GVBCRN # full - T
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model VTBCRN # full-G 
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model GBXDSW # full - V - T
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model GBDSW # full - V - X - T
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model GBTXDSW # full-V 
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model DSW
python train_MADSW.py --data carla --epochs 20 --batch-size 256 --model RNN 

# boid
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model GVTBCRN # full 
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model GVBCRN # full - T
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model GBDSW # full - V - X - T
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model GBTXDSW # full-V
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model GBXDSW # full - V - T
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model VTBCRN # full-G  
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model DSW 
python train_MADSW.py --data boid --epochs 20 --batch-size 256 --model RNN 

# nba 
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model GVTTBCRN --vel --l_X 1 # full
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model VTTBCRN --vel --l_X 1 # -G
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model GBDSW --vel --l_X 1 # full - V - X - T
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model GBXDSW --vel --l_X 1 # full - V - T
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model GBTTXDSW --vel --l_X 1 # full-V
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model GVBCRN --vel --l_X 1 # -TT
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model DSW --vel --l_X 1 
python train_MADSW.py --data nba --n_games 180 --epochs 20 --batch-size 256 --model RNN --vel --l_X 1 

### boid dataset
# cd simulation
# python generate_boid_dataset.py --ver 0 --num-train 20000 --num-valid 400 --num-test 400 --sample_freq 1 --length 21 --length_test 21 --n_boids 20 --r_o 1 # 
