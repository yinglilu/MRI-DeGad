
from nilearn import image
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd
import os


def resampling(gad,nongad,output_gad,output_nongad):
    subject = gad.split("/")[3]
    for input, output in zip((gad,nongad),(output_gad,output_nongad)):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        orig_img=nib.load(input)
        resample = image.resample_img(orig_img, target_affine=np.eye(3), interpolation='linear')
        nib.save(resample, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample")
    parser.add_argument("--gad_input", required=True, help="Input Gad")
    parser.add_argument("--nongad_input", required=True, help="Input Nongad")
    parser.add_argument("--gad_output", required=True, help="Gad Output Directory.")
    parser.add_argument("--nongad_output", required=True, help="Nongad Output Directory.")
  
    args = parser.parse_args()
    input_gad = "../" + args.gad_input
    input_nongad = "../" + args.nongad_input
    output_gad= "../" + args.gad_output
    output_nongad= "../" + args.nongad_output

    resampling(input_gad,input_nongad,output_gad,output_nongad)