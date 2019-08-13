# CheXpert Baseline PyTorch

## 0. Update remark
- 2019.08.07

I am trying to repeat the CheXpert baseline model and doing some research on Pneumonia. Welcome anyone interested in finishing this work to join the project and train and improve the model together!.

- 2019.08.08

The baseline model have been trained. The result is almost same as the paper said.

## 1. How to run the code.

-> run 'main.py'

-> select mission

## 2. Notation
- The model is trained through GTX 1060. if you want to run the code with cpu, you should set 'use_gpu = False' in 'config.py' and if you want to load checkpoint to the model stored in cpu, you should change the code in 'main.py' line 167
- Don't forget to change the 'data_root' in 'config.py' to satisfy the root location where you store CheXpert data.

