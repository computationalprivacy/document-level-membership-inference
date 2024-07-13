#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=32:mem=124gb

module load anaconda3/personal
source activate copyright

cd ~/copyright_inference/src

python import_books.py --write_dir='/rds/general/user/mm422/home/copyright_inference/data/raw/' --hf_dataset="wikitext-103-raw-v1"