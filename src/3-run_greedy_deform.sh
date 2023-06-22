#!/bin/bash
#applying rigid and affine transforms to nongad images to place in gad space

input_gad_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/resampled/gad' # change dirs to make it from CBS perspective since greedy on CBS
input_nongad_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/resampled/nongad'
output_dir='/home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/derivatives/greedy'
sub_list=`cat /home/ROBARTS/fogunsanya/graham/projects/ctb-akhanf/cfmm-bids/Lau/degad/MONAI_scripts/pipeline/0-subject_list.txt`

for sub in ${sub_list}; do
        mkdir ${output_dir}/${sub}
        ##TODO: TEST that i can take away the interpolartion param
        #rigid transform
        greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat -ia-image-centers -n 100x50x10
        #applying rigid transform
        greedy -d 3 -dof 6 -rf ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -rm ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz  ${output_dir}/${sub}/${sub}_acq-nongad_desc-rigid_T1w.nii.gz -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat
        
        #affine transform
        greedy -d 3 -a -dof 12 -m NCC 2x2x2 -i ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat -ia-image-centers -n 100x50x10
        #applying affine transform
        greedy -d 3 -dof 12 -rf ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -rm ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz  ${output_dir}/${sub}/${sub}_acq-nongad_desc-affine_T1w.nii.gz -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat
        
       

        for i in {1..4}; do 
            #need to generate 4 types of deformable registrations with regularization applied using the affine transform
            greedy -d 3 -m NCC 2x2x2 -i ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz -it ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_deform_${i}.mat -sv -n 100x50x10 -s $((i + 1))mm $((i + 0))mm
            
            #need to apply 4 deformable registrations 

            greedy -d 3 -rf ${input_gad_dir}/${sub}/${sub}_acq-gad_resampled_T1w.nii.gz -rm ${input_nongad_dir}/${sub}/${sub}_acq-nongad_resampled_T1w.nii.gz ${output_dir}/${sub}/${sub}_acq-nongad_desc-deform_${i}_T1w.nii.gz -r $ ${output_dir}/${sub}/${sub}_from_nongad_to_gad_deform_${i}.mat
        done
done
