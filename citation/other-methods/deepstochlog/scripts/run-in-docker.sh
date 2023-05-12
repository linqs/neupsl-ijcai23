#!/bin/bash

cp /home/$(whoami)/scripts/run_citation.py /home/$(whoami)/deepstochlog/run_citation.py

cd deepstochlog

for split in 0 1 2 3 4; do
  for dataset_name in citeseer cora; do
    python3 run_citation.py ${dataset_name} ${split} > /home/$(whoami)/results/${dataset_name}/${split}/out.txt 2> /home/$(whoami)/results/${dataset_name}/${split}/out.err
  done
done
