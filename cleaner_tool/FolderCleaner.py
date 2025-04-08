'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\sort\\cleaner_tool\\FolderCleaner.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-09 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `FolderCleaner` class, which provides utility functions for cleaning up directories.
It performs the following tasks:
1. Clears all files and optionally subfolders in a specified directory.
2. Recursively deletes extra folders and files that do not match a list of allowed folder names.
3. Ensures that only specified folders (e.g., `images` and `labels`) and their contents are retained.

The script is designed for managing and cleaning up datasets by removing unnecessary files and folders.
'''

import os
import shutil

class FolderCleaner:
    def __init__(self):
        pass
    
    def clear_folder(self, folder_path, keep_subfolders=True):
        """
        Clear all files and optionally subfolders in the specified directory.

        Args:
            folder_path (str): Path to the folder to be cleaned.
            keep_subfolders (bool): Whether to retain subfolders. Defaults to True.
        """
        if not os.path.exists(folder_path):
            print(f"{folder_path} does not exist.")
            return

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            
            if not keep_subfolders:
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")

    def clear_extra_folders(self, folder_path, allowed_folders=None):
        """
        Recursively delete extra folders and files that do not match the allowed folder list.

        Args:
            folder_path (str): Path to the input folder.
            allowed_folders (list): List of folder names to retain. Defaults to ['images', 'labels'].
        """
        if not os.path.exists(folder_path):
            print(f"{folder_path} does not exist.")
            return

        if allowed_folders is None:
            allowed_folders = ['images', 'labels']

        # Traverse the directory and process each subfolder and file
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Delete the folder if it is not in the allowed list
                if dir_name not in allowed_folders:
                    shutil.rmtree(dir_path)
                    print(f"Deleted extra folder: {dir_path}")

            # Delete files in directories that are not in the allowed list
            for file in files:
                # Get the name of the parent folder
                parent_folder = os.path.basename(root)
                # Delete the file if its parent folder is not in the allowed list
                if parent_folder not in allowed_folders:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted file not in allowed folders: {file_path}")

# Example usage
if __name__ == "__main__":
    cleaner = FolderCleaner()

    # Recursively clean up extra folders and files in the input folder
    input_folder = '/home/carla/yongzhao/thesis/finalevaluation/7towns/30pics/Town01_Opt'
    cleaner.clear_extra_folders(input_folder)
