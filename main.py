import os
from src.nifti_to_smri import process_nifti_files
from src.phenotype_preprocessing import preprocess_phenotype_data
from src.smri_feature_extraction import extract_smri_features
from src.phenotype_dr import reduce_phenotype_dimensionality
from src.data_fusion import fuse_data
from src.smri_classification import classify_smri_data
from src.phenotype_classification import classify_phenotype_data
from src.classification_revised import revised_classification_smri, revised_classification_phenotype, revised_classification_fusion
from src.sitewise_cv import perform_sitewise_cv  
def main():
    """
    Main function to execute the entire ASD detection pipeline.
    """
    print("Starting the ASD Detection Pipeline...")

    # Define file paths
    phenotype_csv = "Phenotypic_V1_0b_preprocessed.csv"
    processed_phenotype_csv = "data_phenotypic.csv"
    nifti_base_dir = "."  # Assuming NIfTI files are in subdirectories of the current directory
    image_output_dir = "data"
    
    # Create output directory for images if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.join(image_output_dir, "Autism"), exist_ok=True)
    os.makedirs(os.path.join(image_output_dir, "NonAutism"), exist_ok=True)

    # --- Step 1: NIfTI to sMRI Conversion ---
    print("\nStep 1: Converting NIfTI files to sMRI images...")
    process_nifti_files(phenotype_csv, nifti_base_dir, image_output_dir)
    print("NIfTI to sMRI conversion complete.")

    # --- Step 2: Phenotype Preprocessing ---
    print("\nStep 2: Preprocessing phenotype data...")
    preprocess_phenotype_data(phenotype_csv, processed_phenotype_csv)
    print("Phenotype data preprocessing complete.")

    # --- Step 3: sMRI Feature Extraction ---
    print("\nStep 3: Extracting features from sMRI images...")
    extract_smri_features(image_output_dir)
    print("sMRI feature extraction complete.")

    # --- Step 4: Phenotype Dimensionality Reduction ---
    print("\nStep 4: Performing dimensionality reduction on phenotype data...")
    reduce_phenotype_dimensionality(processed_phenotype_csv)
    print("Phenotype dimensionality reduction complete.")

    # --- Step 5: Data Fusion ---
    print("\nStep 5: Fusing sMRI and phenotype data...")
    # Using VGG features for fusion as it is used in the sitewise CV
    smri_features_csv = "vgg170.csv"
    fused_data_csv = "merged_vgg_pheno.csv"
    fuse_data(processed_phenotype_csv, smri_features_csv, fused_data_csv)
    print("Data fusion complete.")

    # --- Step 6: sMRI Classification ---
    print("\nStep 6: Performing classification based on sMRI data...")
    classify_smri_data("cnn-170.csv", processed_phenotype_csv)
    classify_smri_data("resnet170.csv", processed_phenotype_csv)
    classify_smri_data("vgg170.csv", processed_phenotype_csv)
    print("sMRI classification complete.")

    # --- Step 7: Phenotype Classification ---
    print("\nStep 7: Performing classification based on phenotype data...")
    classify_phenotype_data("pca20.csv", processed_phenotype_csv)
    print("Phenotype classification complete.")

    # --- Step 8: Final Revised Classification ---
    print("\nStep 8: Performing revised and detailed classification analysis...")
    print("\n--- sMRI Revised Classification (VGG) ---")
    revised_classification_smri("vgg170.csv", processed_phenotype_csv)
    
    print("\n--- Phenotype Revised Classification ---")
    revised_classification_phenotype(processed_phenotype_csv)

    print("\n--- Fusion Revised Classification ---")
    revised_classification_fusion(fused_data_csv, processed_phenotype_csv)
    
    # --- Step 9: Sitewise Cross-Validation (NEW STEP) ---
    print("\nStep 9: Performing Leave-One-Site-Out Cross-Validation...")
    # This analysis uses VGG features as an example.
    # You can change "vgg170.csv" to "cnn-170.csv" or "resnet170.csv" for comparison.
    perform_sitewise_cv(
        phenotype_csv=processed_phenotype_csv,
        features_csv="vgg170.csv",
        output_latex_path="site_majority_table_vgg.tex"
    )
    print("Sitewise cross-validation complete.")
    
    print("\nASD Detection Pipeline finished.")

if __name__ == "__main__":

    main()
