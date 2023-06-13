#!/bin/bash
#applying fcm normalization to the nongad image
input_nongad_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/greedy'
fcm_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/normalized_fcm'
sub_list=`cat /project/6050199/akhanf/cfmm-bids/data/Lau/degad/MONAI_scripts/pipeline/0-subject_list.txt`

for sub in ${sub_list}; do
        #applying fcm-normalize, using rigidly transformed WM mask
        mkdir ${fcm_dir}/${sub}
        fcm-normalize ${input_nongad_dir}/${sub}/${sub}_acq-nongad_desc-rigid_T1w.nii.gz -o ${fcm_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz -mo t1 -p -v -tt wm -tm ${input_nongad_dir}/${sub}/${sub}_nongad_WM_seg_gad_space.nii.gz 

        done