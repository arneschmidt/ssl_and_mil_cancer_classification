# Camelyon16 dataset scripts
These scripts were used to preprocess the Camelyon16 dataset.

To reproduce the preprocessing, please follow the following steps:

1. Install the environment with anaconda/miniconda (use the environment.yaml of this folder!): 
    * `conda env create --file environment.yaml`
    * `conda activate tensorlfow_2_3`
2. Run the script get get_camelyon16_masks.py , defining input and output folder
    * `python get_camelyon16_masks.py -h` to see the arguments
3. Run the script get wsi_to_patches.py , defining input and output folder
    * `python get wsi_to_patches.py -h` to see the arguments