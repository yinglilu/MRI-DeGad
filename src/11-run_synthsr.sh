#!/bin/bash
#need to run as regular interactive or as batch script : regularInteractive -n 32 -m 16000 -t 24

#move fmriprep gad images into one folder

gad_bids_path_1='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/gad' #subs 1-55
gad_bids_path_2='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep/gad_55_63' # subs 55-63
all_gads='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/all_gads'
output_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/synthsr/all'
synth_bids='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/synthsr/bids'
sub_list=`cat /project/6050199/akhanf/cfmm-bids/data/Lau/degad/MONAI_scripts/pipeline/0-subject_list.txt`
synth_cont='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/containers/synthsr_main.sif'


for dir in $(ls -d ${gad_bids_path_1}/sub-????); do
    sub=$(basename "$dir")
    cp ${gad_bids_path_1}/${sub}/ses-pre/anat/${sub}_ses-pre_acq-gad_run-1_desc-preproc_T1w.nii.gz ${all_gads}/
done

for dir in $(ls -d ${gad_bids_path_2}/sub-????); do
    sub=$(basename "$dir")
    cp ${gad_bids_path_2}/${sub}/ses-pre/anat/${sub}_ses-pre_acq-gad_run-1_desc-preproc_T1w.nii.gz ${all_gads}/
done

#run synthsr command

singularity run ${synth_cont} ${all_gads}/ ${output_dir}/ --cpu --threads 32

#move all output synthsr images into bids format
for sub in $sub_list; do
    mkdir -p ${synth_bids}/${sub}/ses-pre/anat
    cp ${output_dir}/${sub}_ses-pre_acq-gad_run-1_desc-preproc_T1w_SynthSR.nii.gz ${synth_bids}/${sub}/ses-pre/anat/ 
    mv ${synth_bids}/${sub}/ses-pre/anat/${sub}_ses-pre_acq-gad_run-1_desc-preproc_T1w_SynthSR.nii.gz ${synthsr_bids}/${sub}/ses-pre/anat/${sub}_ses-pre_acq-gad_run-1_desc-preproc_SynthSR_T1w.nii.gz # need to rename to make fmriprep know it is a t1w image
done

