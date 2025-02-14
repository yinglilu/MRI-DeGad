# MRI-DeGad
Convolutional Neural Network for the Conversion of Gadolinium-Enhanced T1-weighted MRI to Non-Gadolinium T1w Scans

## Directory of outputs
The CNN model outputs are stored at `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output`

Checkpoint is stored at `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output/<model>/checkpoint.pt`

Output niftii from a test subject is stored at `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output/<model>/test`

## Setup
  1. Ensure you are logged onto graham. To clone the MRI-degad repo onto your scratch run:
  `git clone git@github.com:fogunsan/MRI-DeGad.git` or `git clone https://github.com/fogunsan/MRI-DeGad.git`
  2. A venv can be created using kslurm with the following command:
  `kpy create venv_degad`
  3. Following venv creation, cd into the cloned directory and run:
  `pip install -r venv_requirements.txt`

## Running preprocessing snakemake workflow (TBA)

## Running  CNN/GAN snakemake model training workflow
  16 mm and 32 mm pregenerated patches (outputs from preprocessing pipeline) are stored at `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/derivatives/patches`.
  Can run snakemake model training workflow to output model checkpoints and output images in `output/` directory of `snakemake_CNN` or `snakemake_GAN` directory.
  Can vary arguments at top of Snakefile ensuring max combinations in a single job does not surpass 4 due to time limits.
  Search space for CNN:
      Patch size: [15,31]
      Batch size: [32,64,128]
      Learning rate: [0.01,0.001,0.005,0.0001]
      Initial number of filters: [16,32,64]
      Number of Convolutions: [2,3]
      Loss: [MAE]
  1. cd into cloned repo.
  2. `cd snakemake_CNN` or `cd snakemake_GAN`, depending on what model you are training
  3. `mkdir output` if it does not already exist
  4. Adjust arguments at top of Snakefile. Can check `/project/6050199/akhanf/cfmm-bids/data/Lau/shared/training/snakemake_CNN/output` to ensure combination has not already been tested
  5. In `scripts/run_training.sh` file, change repo variable to one located on your scratch.
  6. `bash scripts/run_training.sh`


## Running inference on trained MRI-degad models on graham
  1. Sufficient computational resources must be requested. Sample command:
  `regularInteractive -n 16 -m 64000 -t 1 -g`
  
  2. The venv can then be loaded:
  `kpy load venv_train_degad`

  Checkpoints from trained MRI-degad-CNN models are located at:
  `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output`

  Script to run inference on a directory of gad images is located at:
  `/project/6050199/akhanf/cfmm-bids/data/Lau/degad/shared/inference/inference_degad_CNN.py`
    
  3. To run inference script:
  `cd /project/6050199/akhanf/cfmm-bids/data/Lau/degad/shared/inference`
  `python3 inference_degad_CNN.py --checkpoint /project/6050199/akhanf/cfmm-bids/data/Lau/shared/training/snakemake_CNN/output/<model>/checkpoint.pt --gad_direc <input_gad_dir> --output_dir <output_degad_dir> --degad_ds`

  **Note that directory of gad images must be in non-bids format and that the degad output directory is in bids format
  ** using the optional --degad_ds flag indicates that bids files (CHANGES,  scans.json,  dataset_description.json, participants.json, README) from original seeg/dbs degad dataset (`/project/6050199/akhanf/cfmm-bids/data/Lau/degad/bids`) will be output to make output directory fmriprep ready. If using an external dataset, need to manually input those files afterwards for fMRIPrep to run. **

  sample command:
  `python3 inference_degad_CNN.py --checkpoint /project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output/patch-16_batch-128_LR-0.001_filter-32_depth-3_convs-2_loss-mae/checkpoint.pt --gad_direc ../../derivatives/test_set/ --output_dir ../../derivatives/test_set/test_inference/`
  


## Pipeline

Path to degad non-gad and gad bids directory on Graham: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/bids  
**Preprocessing code still has my scratch directories, will update to refer to project directory

**0-Subject_list.txt**  
  What it does: Holds all 55 subjects that underwent DBS or SEEG at LHSC

**1-running fMRIPrep**  
  What it does: Run fMRIPrep on the gad and nongad bids dataset. Outputs have N4 bias correction applied  
  
  Terminal command:  
  bidsBatch -j Long fmriprep_21.0.0 /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/degad_bids_gad /home/fogunsan/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/fmriprep/gad participant --anat-only --skip_bids_validation --omp-nthreads 8 --nprocs 16 
  
  Input directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/degad_bids_gad and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/degad_bids_nongad  
  
  Output directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/gad and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/nongad

**2-isotropic_resampling.ipynb**  
What it does: Resamples images to 1 mm volumetric resolution    

 Input directories:  /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/gad and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/nongad    
 
Output directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/resampled/gad and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/resampled/nongad  

**3-run_greedy.sh**   
What it does: Performs a rigid and affine transformation, putting nongad images to gad space   

Input directories: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/resampled/gad and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/resampled/nongad  

Output directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/greedy

**4-registration_QC.ipynb**   
What it does: Outputs html file to perform qualitative assessment of the registrations (with preference for rigid registrations) and exclude subjects with pathology  

Input directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/greedy  

Output QC file: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/registration_QC_0.html  

QC Google Sheets: https://docs.google.com/spreadsheets/d/19TnoOD47vY6vuqSzujC0RgwMGdMhtJWCleeeG325nNo/edit?usp=sharing  

Passing subject directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/
  
**5-run_fcm_whitestripe_norm.sh**  
What it does: Fuzzy c means normalization normalizes on a subject level the nongad image to the mean of the white matter (using a fMRIPrep WM segmentation mask) which is scaled to 1. Whitestripe normalization performs Z-score normalization based on the intensity values of normal appearing white matter  

Input directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/  

Command to download library: pip install intensity-normalization (https://github.com/jcreinhold/intensity-normalization)  

Output directories: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/normalized_fcm/ and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/normalized_ws/
  
  
**6a-train_degad_CNN.ipynb**  
What it does: File to train degad model using a convolutional neural network (U-Net) implementation.  

Input directories: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/ and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/normalized_fcm/  

Output directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/UNET_outputs/
  
Command to run:  
1. convert to python script: jupyter nbconvert testnotebook.ipynb --to python  
2. kbatch gpu -j Long --venv venv_train_degad python 6a-train_degad_CNN.py  

**6b-train_degad_GAN.ipynb**  
What it does: File to train degad model using a generative adversarial network (GAN) implementation  

Input directory: /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/ and /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/normalized_fcm/  

Output directory:/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/GAN_outputs/
  
Command to run:  
1. convert to python script: jupyter nbconvert testnotebook.ipynb --to python  
2. kbatch gpu -j Long --venv venv_train_degad python 6b-train_degad_GAN.py
