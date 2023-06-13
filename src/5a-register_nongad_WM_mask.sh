#!/bin/bash
#applying greedy transform on WM-probability mask to be used in the fcm normalization

input_mask_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/fmriprep' # change dirs to make it from CBS perspective since greedy on CBS
input_nongad_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/greedy'
output_mask_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/greedy'
sub_list=`cat /home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/MONAI_scripts/pipeline/0-subject_list.txt`


for sub in ${sub_list}; do
    greedy -d 3 -rf ${input_nongad_dir}/${sub}/${sub}_acq-nongad_desc-rigid_T1w.nii.gz -rm ${input_mask_dir}/nongad*/${sub}/ses-pre/anat/${sub}_ses-pre_acq-nongad_run-1_label-WM_probseg.nii.gz ${output_mask_dir}/${sub}/${sub}_nongad_WM_seg_gad_space.nii.gz -r ${input_nongad_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat

done