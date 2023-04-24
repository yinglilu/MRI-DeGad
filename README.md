# MRI-DeGad
Convolutional Neural Network for the Conversion of Gadolinium-Enhanced T1-weighted MRI to Non-Gadolinium T1w Scans using a CNN and a GAN


## Pipeline

Path to raw non-gad and gad bids directory:________

0-Subject_List
  What it does: Holds all 55 subjects that underwent DBS or SEEG at LHSC

1- running fMRIPrep
  Ran fMRIPrep on the gad and nongad bids dataset. Outputs have N4 bias correction applied
  Terminal command: 
  Input directory: 
  Output directory:

2-isotropic_resampling.ipynb
  What it does: Resamples 
  Input directory:
  Output directory:
  
3-run_greedy.sh
  What it does: Performs a rigid and affine transformation, taking nongad images to gad space
  Input directory:
  Output directory:

4-registration_QC.ipynb
  What it does: Outputs html file to perform qualitative assessment of the registrations (with preference for rigid registrations) and exclude subjects with pathology
  Input directory:
  Output directory:
  QC Google Sheets:

5a-train_degad_CNN.ipynb
  What it does: File to train degad model using a convolutional neural network (U-Net) implementation
  Input directory:
  Output directory:
  
5b-train_degad_GAN.ipynb
  What it does: File to train degad model using a generative adversarial network (GAN) implementation
  Input directory:
  Output directory:
