#!/bin/bash
#running fmriprep version 21.0.0 on gad and nongad bids directory

input_gad_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/degad_bids_gad'
input_nongad_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/degad_bids_nongad'
output_dir='/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/fmriprep'

bidsBatch -j Long fmriprep_21.0.0 ${input_gad_dir} ${output_dir}/gad participant --anat-only --skip_bids_validation --omp-nthreads 8 --nprocs 16

bidsBatch -j Long fmriprep_21.0.0 ${input_nongad_dir} ${output_dir}/nongad participant --anat-only --skip_bids_validation --omp-nthreads 8 --nprocs 16