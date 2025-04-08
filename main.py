'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\sort\\main.py
Author       : Yongzhao Chen
Date         : 2024-09-09 12:20:35
LastEditTime : 2024-09-09 12:23:17
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
This script processes subfolders within a specified top-level directory. 
It performs the following steps for each subfolder:
1. Cleans up unnecessary files and folders using the `FolderCleaner` class.
2. Processes the cleaned folder using the `FullPipelineProcessor` class, for detail info plz check the description in `FullPipelineProcessorTowns.py` file.

The script is designed for batch processing of multiple subfolders in a directory.
'''
import os
from FullPipelineProcessorTowns import FullPipelineProcessor
from cleaner_tool.FolderCleaner import FolderCleaner

if __name__ == "__main__":
    # Define the top-level directory
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/30pics'
    config_path = 'config/config.yaml'

    # Initialize the FolderCleaner instance
    cleaner = FolderCleaner()

    # Get a list of all subfolders in the top-level directory
    folders_list = [f for f in os.listdir(top_level_layer) if os.path.isdir(os.path.join(top_level_layer, f))]
    # print(f"Found {len(folders_list)} subfolders in {top_level_layer}.")
    # print(folders_list)

    for folder_name in folders_list:
        input_folder_path = os.path.join(top_level_layer, folder_name)
        print(f"Processing folder: {input_folder_path}")
        # 1. Clean the current subfolder (e.g., remove unnecessary files and folders)
        cleaner.clear_extra_folders(input_folder_path)
        # 2. Process each subfolder
        processor = FullPipelineProcessor(input_folder_path, config_path)
        processor.run()

    print("All folders processed.")
