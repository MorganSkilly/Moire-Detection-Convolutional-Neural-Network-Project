import os
import shutil

# Define the source directory
source_dir = 'C:/Users/1903578/OneDrive - Abertay University/Dataset/train/train'

# Define the destination directories
clean_dir = 'C:/Users/1903578/OneDrive - Abertay University/Dataset/train/train/clean'
moire_dir = 'C:/Users/1903578/OneDrive - Abertay University/Dataset/train/train/moire'

# Iterate through each file and subdirectory in the source directory
for root, dirs, files in os.walk(source_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        
        # Check if the file name contains '_gt'
        if '_gt' in filename:
            # Move the file to the clean directory and rename it
            new_filename = filename.replace('_gt', '')
            new_file_path = os.path.join(clean_dir, new_filename)
            shutil.move(file_path, new_file_path)
            
        # Check if the file name contains '_moire'
        elif '_moire' in filename:
            # Move the file to the moire directory and rename it
            new_filename = filename.replace('_moire', '')
            new_file_path = os.path.join(moire_dir, new_filename)
            shutil.move(file_path, new_file_path)
            
        # Remove the original file
        os.remove(file_path)
