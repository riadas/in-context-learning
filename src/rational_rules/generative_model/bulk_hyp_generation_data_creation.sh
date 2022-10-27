#!.bin/bash

for entry in "$1"/*
do
  echo "$entry"
  python ../../../generative_model/create_existing_data_to_hyp_generation.py $entry
  #python ../../../generative_model/create_existing_data_to_hyp_generation.py $entry missing-and-or
  #python ../../../generative_model/create_existing_data_to_hyp_generation.py $entry all
done
