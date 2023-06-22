#!/bin/bash
# moving nongad images from fcm dir into passing dir 
# moving gad images from gresampled dir into passing dir

input_nongad_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/normalized_fcm' # change dirs to make it from CBS perspective since greedy on CBS
input_gad_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/resampled/gad'
output_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/passing_dataset'
sub_list=`cat /home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/passing_dataset/passing_dataset.txt`


for sub in ${sub_list}; do
    mkdir ${output_dir}/${sub}
    cp ${input_nongad_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${output_dir}/${sub}/
  
done
