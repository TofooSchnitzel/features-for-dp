# Fixed Feature Extractors for Differentially Private Machine Learning on Medical Imaging

## Prerequisits
- Install conda

- Download the paediatric pneumonia dataset 
  - https://github.com/gkaissis/PriMIA/tree/400bdf6a9b6b987520e98f327600ad8be2e556c0/data
  - You need the folders test and train, as well as the files Chest_xray_Corona_Metadata.csv and Labels.csv
  - The following folder structure is expected: data/raw/pppp, specify your path in the config (without pppp)

- Download the PrivateGBDT: You need only the folder federated_gbdt to the folder src in this repository
  - https://github.com/Samuel-Maddock/federated-boosted-dp-trees/tree/2a8017e7699ab139bffd58287743abe279093eb1/federated_gbdt

- To use the pre-trained models, the weights are required


## Installation
- Install the required dependencies with conda env create -f environment.yaml
- Activate the environment with conda activate objaxdp
- Install with pip: pip install --upgrade jax[cuda]==0.3.15 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip install objax==1.6.0
