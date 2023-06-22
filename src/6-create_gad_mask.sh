#!/bin/bash
# file to run c3d in graham on all subjects in passing dir using the nongad image as a brain mask
input_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset' #we are skull stripping gad images that have been resampled to create maskd
synthstrip_dir='/scratch/fogunsan/containers/synthstrip.1.3.sif'
sub_list=`cat /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/passing_dataset.txt`
output_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/synthstrip'

for sub in ${sub_list}; do
    mkdir ${output_dir}/${sub}
    singularity run ${synthstrip_dir} -i ${input_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -o ${output_dir}/${sub}/${sub}_acq-gad_stripped_T1w.nii.gz -m ${output_dir}/${sub}/${sub}_acq-gad_mask.nii.gz -b 3

done