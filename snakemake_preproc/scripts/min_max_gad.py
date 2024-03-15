
import numpy as np
import nibabel as nib
import argparse


def normalize(image_path,output_dir):
    print(image_path,str(image_path).split("/"), len(str(image_path).split("/")))
    sub = image_path.split("/")[3]

    image_nifti = nib.load(image_path)
    image_array = image_nifti.get_fdata()
    scaled_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    scaled_image_nifti = nib.Nifti1Image(scaled_image, image_nifti.affine, image_nifti.header)
    output_path = f'{output_dir}/{sub}/{sub}_acq-gad_rescaled_T1w.nii.gz' 
    nib.save(scaled_image_nifti, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Min Max Normalize")
    parser.add_argument("--input", required=True, help="Input File")
    parser.add_argument("--output", required=True, help="Output Directory.")
  
    args = parser.parse_args()
    input_file = "../" + args.input
    output= "../" + args.output

    normalize(input_file,output)