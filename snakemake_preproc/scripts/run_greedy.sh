
#!/bin/bash

input_nongad="$1"
input_gad="$2"
rigid_nongad="$3"
affine_nongad="$4"


sub=$(echo $input_nongad | awk -F'/' '{print $(NF-1)}')  
output_dir=$(echo "$rigid_trans" | rev | cut -d'/' -f3- | rev)



mkdir ${output_dir}/${sub}
#rigid transform
greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i ${input_gad} ${input_nongad} -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat -ia-image-centers -n 100x50x10
#applying rigid transform
greedy -d 3 -dof 6 -rf ${input_nongad} -rm ${input_nongad} ${rigid_nongad} -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_rigid.mat

#affine transform
greedy -d 3 -a -dof 12 -m NCC 2x2x2 -i ${input_gad} ${input_nongad} -o ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat -ia-image-centers -n 100x50x10
#applying affine transform
greedy -d 3 -dof 12 -rf ${input_nongad} -rm ${input_nongad}  ${affine_nongad} -r ${output_dir}/${sub}/${sub}_from_nongad_to_gad_affine.mat


