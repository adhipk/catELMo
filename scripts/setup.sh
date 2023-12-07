module purge
module load mamba/latest
mamba create --name bap python=3.6.13
pip install pandas==1.1.5 tensorflow==2.6.0 keras==2.6.0 scikit-learn==0.24.2 tqdm