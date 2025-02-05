import os
import shutil

def copy_and_rename_files(src_parent_folder, dst_folder):
    # Ensure the target folder exists
    os.makedirs(dst_folder, exist_ok=True)
    
    # Create 'clean' and 'moire' subfolders in the destination folder
    clean_folder = os.path.join(dst_folder, "clean")
    moire_folder = os.path.join(dst_folder, "moire")
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(moire_folder, exist_ok=True)

    clean_counter = 0  # For renaming clean images
    moire_counter = 0  # For renaming moire images

    # Iterate through each group folder
    for group_name in os.listdir(src_parent_folder):
        group_path = os.path.join(src_parent_folder, group_name)

        # Skip non-folder items
        if not os.path.isdir(group_path):
            continue

        clean_path = os.path.join(group_path, 'clean')
        moire_path = os.path.join(group_path, 'moire')

        # Check if 'clean' subfolder exists in the current group
        if os.path.isdir(clean_path):
            for file_name in os.listdir(clean_path):
                src_file_path = os.path.join(clean_path, file_name)
                if os.path.isfile(src_file_path):
                    clean_counter += 1
                    new_file_name = f"{str(clean_counter).zfill(4)}_clean.jpg"  # Create unique name
                    dst_file_path = os.path.join(clean_folder, new_file_name)
                    shutil.copy(src_file_path, dst_file_path)
                    print(f"Copied {src_file_path} to {dst_file_path}")

        # Check if 'moire' subfolder exists in the current group
        if os.path.isdir(moire_path):
            for file_name in os.listdir(moire_path):
                src_file_path = os.path.join(moire_path, file_name)
                if os.path.isfile(src_file_path):
                    moire_counter += 1
                    new_file_name = f"{str(moire_counter).zfill(4)}_moire.jpg"  # Create unique name
                    dst_file_path = os.path.join(moire_folder, new_file_name)
                    shutil.copy(src_file_path, dst_file_path)
                    print(f"Copied {src_file_path} to {dst_file_path}")

# Example usage:
src_parent_folder = "Dataset/train/train"  # Replace with your parent folder path (e.g., group_1, group_2, etc.)
dst_folder = "Dataset/train/train/all"  # Replace with the destination folder path (e.g., allgroups)
copy_and_rename_files(src_parent_folder, dst_folder)
