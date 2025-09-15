import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

def get_label_with_id(df, num):
    """
    Retrieves the site ID and diagnosis group for a given subject ID.

    Args:
        df (pd.DataFrame): The phenotype dataframe.
        num (int): The subject ID.

    Returns:
        tuple: A tuple containing the formatted subject identifier (SITE_ID_subject)
               and the diagnosis group (1 for Autism, 2 for Non-Autism).
    """
    for a, i in enumerate(df['SITE_ID']):
        b = df.iloc[a]['subject']
        if num == b:
            return f"{i}_{b}", df.iloc[a]['DX_GROUP']
    return None, None

def save_nifti_slice_as_png(path, num, df, output_dir):
    """
    Loads a NIfTI file, extracts a slice, and saves it as a PNG image in the
    appropriate directory (Autism/NonAutism) based on the diagnosis group.

    Args:
        path (str): The path to the NIfTI file.
        num (int): The subject ID.
        df (pd.DataFrame): The phenotype dataframe.
        output_dir (str): The base directory to save the images.
    """
    nifti_img = nib.load(path)
    nifti_data = nifti_img.get_fdata()

    middle_slice = nifti_data.shape[2] // 2
    
    label, dx_group = get_label_with_id(df, num)
    
    if label is None:
        print(f"Warning: Subject ID {num} not found in phenotype data.")
        return

    if dx_group == 1:
        save_dir = os.path.join(output_dir, "Autism")
    else:
        save_dir = os.path.join(output_dir, "NonAutism")
        
    plt.imshow(nifti_data[:, :, middle_slice], cmap='BrBG')
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{label}.png"), dpi=100)
    plt.close()

def process_nifti_files(phenotype_csv, base_dir, output_dir):
    """
    Walks through the directory structure to find anatomical NIfTI files,
    and processes them to generate sMRI images.

    Args:
        phenotype_csv (str): Path to the phenotype CSV file.
        base_dir (str): The base directory containing the NIfTI files.
        output_dir (str): The directory to save the output PNG images.
    """
    df = pd.read_csv(phenotype_csv)

    for dirpath, _, filenames in os.walk(base_dir):
        if "anat" in dirpath:
            for filename in filenames:
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    full_path = os.path.join(dirpath, filename)
                    try:
                        # Extract subject ID from the directory path
                        subject_id_str = [part for part in dirpath.split(os.sep) if part.startswith('sub-')]
                        if subject_id_str:
                            subject_id = int(subject_id_str[0].split('-')[-1])
                            save_nifti_slice_as_png(full_path, subject_id, df, output_dir)
                            print(f"Processed: {full_path}")
                    except Exception as e:
                        print(f"Could not process {full_path}: {e}")