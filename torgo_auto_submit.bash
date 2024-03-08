#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p short
#SBATCH --time=08:00:00
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2022.05
module load cuda/11.7

source activate /work/van-speech-nlp/jindaznb/asrenv/




## torgo expriement

# python tauto.py --test_speaker F01 --split test --level word --pattern keep_all
# python tauto.py --test_speaker F03 --split test --level word --pattern keep_all
# python tauto.py --test_speaker F04 --split test --level word --pattern keep_all
# python tauto.py --test_speaker M01 --split test --level word --pattern keep_all
# python tauto.py --test_speaker M02 --split test --level word --pattern keep_all
# python tauto.py --test_speaker M03 --split test --level word --pattern keep_all
# python tauto.py --test_speaker M04 --split test --level word --pattern keep_all
# python tauto.py --test_speaker M05 --split test --level word --pattern keep_all

python tauto.py --test_speaker F01 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker F03 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker F04 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker M01 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker M02 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker M03 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker M04 --split test --level sentence --pattern keep_all
python tauto.py --test_speaker M05 --split test --level sentence --pattern keep_all

# python tauto.py --test_speaker F01 --split test --level word --pattern no_keep
# python tauto.py --test_speaker F03 --split test --level word --pattern no_keep
# python tauto.py --test_speaker F04 --split test --level word --pattern no_keep
# python tauto.py --test_speaker M01 --split test --level word --pattern no_keep
# python tauto.py --test_speaker M02 --split test --level word --pattern no_keep
# python tauto.py --test_speaker M03 --split test --level word --pattern no_keep
# python tauto.py --test_speaker M04 --split test --level word --pattern no_keep
# python tauto.py --test_speaker M05 --split test --level word --pattern no_keep

# python tauto.py --test_speaker F01 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker F03 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker F04 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker M01 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker M02 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker M03 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker M04 --split test --level sentence --pattern no_keep
# python tauto.py --test_speaker M05 --split test --level sentence --pattern no_keep




# python tauto.py --test_speaker F01 --split test --level word --pattern new
# python tauto.py --test_speaker F03 --split test --level word --pattern new
# python tauto.py --test_speaker F04 --split test --level word --pattern new
# python tauto.py --test_speaker M01 --split test --level word --pattern new
# python tauto.py --test_speaker M02 --split test --level word --pattern new
# python tauto.py --test_speaker M03 --split test --level word --pattern new
# python tauto.py --test_speaker M04 --split test --level word --pattern new
# python tauto.py --test_speaker M05 --split test --level word --pattern new

# python tauto.py --test_speaker F01 --split test --level sentence --pattern new
# python tauto.py --test_speaker F03 --split test --level sentence --pattern new
# python tauto.py --test_speaker F04 --split test --level sentence --pattern new
# python tauto.py --test_speaker M01 --split test --level sentence --pattern new
# python tauto.py --test_speaker M02 --split test --level sentence --pattern new
# python tauto.py --test_speaker M03 --split test --level sentence --pattern new
# python tauto.py --test_speaker M04 --split test --level sentence --pattern new
# python tauto.py --test_speaker M05 --split test --level sentence --pattern new