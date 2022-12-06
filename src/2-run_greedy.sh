#!/bin/bash
#applying rigid and affine transforms to nongad images to place in gad space

input_gad_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/resampled/gad'
input_nongad_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/resampled/nongad'
output_dir='/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/greedy'
sub_list=`cat /home/ROBARTS/fogunsanya/graham/scratch/degad/scripts/0-subject_list.txt`

for sub in ${sub_list}; do
        mkdir ${output_dir}/${sub}
        #rigid transform
        greedy -d 3 -a -dof 6 -m NCC 4x4x4 -i ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat -ia-image-centers -n 100x50x0
        #applying rigid transform
        greedy -d 3 -dof 6 -rf ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -rm ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz  ${output_dir}/${sub}/${sub}_acq-nongad_desc-rigid_resliced_T1w.nii.gz -ri LABEL 0.2vox -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat
        
        #affine transform
        greedy -d 3 -a -dof 12 -m NCC 4x4x4 -i ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat -ia-image-centers -n 100x50x0
        #applying affine transform
        greedy -d 3 -dof 12 -rf ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -rm ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz  ${output_dir}/${sub}/${sub}_acq-nongad_desc-affine_resliced_T1w.nii.gz -ri LABEL 0.2vox -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat
        
        #copying resampled gad image into greedy directory
        cp ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${output_dir}/${sub}
    done