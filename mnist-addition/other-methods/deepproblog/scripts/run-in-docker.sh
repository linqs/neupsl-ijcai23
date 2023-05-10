#!/bin/bash

ln -s /home/cfpryor/results/ deepproblog/src/deepproblog/examples/MNIST/results

cd deepproblog/src/deepproblog/examples/MNIST/

python3 run_splits.py > results/out.txt 2> results/out.err
python3 parse_results.py > results/results_out.txt 2> results/results_out.err
mv results.csv results/
