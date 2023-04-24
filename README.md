# MRI-DeGad
Convolutional Neural Network for the Conversion of Gadolinium-Enhanced T1-weighted MRI to Non-Gadolinium T1w Scans using a CNN and a GAN


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
