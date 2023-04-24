#!/bin/bash
#2. applying fcm normalization to the nongad image
#3. applying NAWM normalization to the nongad image
input_nongad_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/passing_dataset'
fcm_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/normalized_fcm'
whitestrip_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/normalized_ws'
sub_list=`cat /home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/passing_dataset/passing_dataset.txt`

for sub in ${sub_list}; do
        #applying fcm-normalize
        fcm-normalize ${input_nongad_dir}/${sub}/${sub}_acq-nongad_desc-rigid_resliced_T1w.nii.gz -o ${fcm_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz -mo t1 -p -v -tt wm -tm ${fcm_dir}/${sub}/${sub}_nongad_WM_seg_gad_space.nii.gz

        # applying whitestrip normalize
        ws-normalize ${input_nongad_dir}/${sub}/${sub}_acq-nongad_desc-rigid_resliced_T1w.nii.gz -o ${whitestrip_dir}/${sub}/${sub}_acq-nongad_normalized_ws.nii.gz -mo t1 -p -v -m ${whitestrip_dir}/${sub}/${sub}_nongad_brain_mask_gad_space.nii.gz 
    done