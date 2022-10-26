#!/bin/bash

bits=(mixed) # 3 5 7 mixed 
examples=(2) # 1 2 
dataset_sizes=(200000) # 50000 100000 200000 
error_probs=(0) # 0 0.01 0.05

for bit in ${bits[@]}
do
  for example in ${examples[@]}
  do
    for dataset_size in ${dataset_sizes[@]}
    do
      for error_prob in ${error_probs[@]}
      do 
        echo "bits: $bit, num_examples_mode: $examples, dataset_size: $dataset_size, error_prob: $error_prob "
        nohup python bulk_data_generation.py $bit $example $dataset_size $error_prob > LOGS/bits.$bit.num_examples_mode.$example.dataset_size.$dataset_size.error_prob.$error_prob.out &
      done
    done  
  done
done
