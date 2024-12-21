#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH -A lp_joris_vermeesch
#SBATCH -J fs
#SBATCH -o fs.out
#SBATCH -e fs.err
#SBATCH --mail-user=antoine.passemiers@kuleuven.be
#SBATCH --mail-type=END,FAIL

# Compilation
cd /lustre1/project/stg_00019/research/Antoine/misc/Feature-Selection-Benchmark
conda activate fs

python real-data-benchmark.py
#python nn-hpo.py
