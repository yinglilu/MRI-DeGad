#!/bin/bash
# file to run c3d in graham on all subjects in passing dir using the nongad image as a brain mask
input_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset'
sub_list=`cat /project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/passing_dataset.txt`
output_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/c3d'

for sub in ${sub_list}; do

    # run c3d using nongad as brain mask and output the number of patches into a text file
    # append patch file to the  training file if sub number is less than 33
    # append patch file to the validation file if sub number is greater than 32 and less than 44
    mkdir ${output_dir}/${sub}
    singularity run /project/6050199/akhanf/singularity/pyushkevich_itksnap_latest.sif c3d ${input_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz ${input_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz -xpa 5 10 -xp ${output_dir}/${sub}/${sub}_samples_31.dat 15x15x15 150| tail -n 1| awk -F':' '{print $NF}' >>  ${output_dir}/num_patches_31.txt 
    
    if [[ "$sub" < "sub-P033" ]]; then
    cat ${output_dir}/${sub}/${sub}_samples_31.dat >> ${output_dir}/training_samples_31.dat

    elif [[ "$sub" > "sub-P032" && "$sub" < "sub-P044" ]]; then
    cat ${output_dir}/${sub}/${sub}_samples_31.dat >> ${output_dir}/validation_samples_31.dat
    fi


done


for sub in ${sub_list}; do

    singularity run /project/6050199/akhanf/singularity/pyushkevich_itksnap_latest.sif c3d ${input_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz ${input_dir}/${sub}/${sub}_acq-nongad_normalized_fcm.nii.gz -xpa 5 10 -xp ${output_dir}/${sub}/${sub}_samples_15.dat 7x7x7 75| tail -n 1| awk -F':' '{print $NF}' >>  ${output_dir}/num_patches_15.txt 
    

    if [[ "$sub" < "sub-P033" ]]; then
    cat ${output_dir}/${sub}/${sub}_samples_15.dat >> ${output_dir}/training_samples_15.dat

    elif [[ "$sub" > "sub-P032" && "$sub" < "sub-P044" ]]; then
    cat ${output_dir}/${sub}/${sub}_samples_15.dat >> ${output_dir}/validation_samples_15.dat
    fi


done
