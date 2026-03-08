# MedSAM3 - Impact 

## Usage 

**Inference**
- infer_sam.py is the proper script for running inference

**Training**
- train_sam3_lora_native.py is the most complete training script

We have two options for training on top of medsam3
1. edit train_sam3_lora_native.py inject the medsam3 LoRA weights after random initilization and continue on those \/
2. merge medsam3 weights into a new sam3 base and then run training script on new base  (prop better?)
