# NoiseFolderFinder
# 用于查找包含 overall_evaluation_results.csv 文件的文件夹，并根据 bias 的范围查找符合条件的文件夹路径。
import os
import pandas as pd

class NoiseFolderFinder:
    def __init__(self, top_level_layer):
        self.top_level_layer = top_level_layer
        self.csv_files = []

    def find_all_csv(self):
        """遍历子文件夹，查找 results 文件夹下的 overall_evaluation_results.csv 文件"""
        for root, dirs, files in os.walk(self.top_level_layer):
            if 'results' in dirs:
                csv_path = os.path.join(root, 'results', 'overall_evaluation_results.csv')
                if os.path.exists(csv_path):
                    self.csv_files.append(csv_path)
        print(f"Found {len(self.csv_files)} CSV files.")

    def parse_gaussian_folder(self, params):
        """解析 Gaussian 参数，返回文件夹路径"""
        mean_value = params.split("mean=")[-1].split(",")[0].strip()
        sigma_value = params.split("sigma=")[-1].split(",")[0].strip()
        folder_name = f"mean{mean_value}_sigma{sigma_value}"
        return f"with_noise/gaussian/{folder_name}"

    def parse_mosaic_folder(self, params):
        """解析 Mosaic 参数，返回文件夹路径"""
        num_value = params.split("num_mosaics=")[-1].split(",")[0].strip()
        color_mode = params.split("color_mode=")[-1].split(",")[0].strip()
        folder_name = f"num{num_value}_{color_mode}"
        return f"with_noise/mosaic/{folder_name}"

    def generate_folder_path(self, base_path, noise_type, params):
        """根据噪声类型生成文件夹路径"""
        if noise_type == "gaussian":
            folder_path = self.parse_gaussian_folder(params)
        elif noise_type == "mosaic":
            folder_path = self.parse_mosaic_folder(params)
        else:
            folder_path = "with_noise/raw"
        return os.path.join(base_path, folder_path)

    def find_folders_by_bias_range(self, min_bias, max_bias):
        """
        查找 bias 值在给定范围内的文件夹路径，并返回这些路径列表。
        
        Args:
            min_bias (float): bias 的下限。
            max_bias (float): bias 的上限。
        
        Returns:
            List[str]: 满足条件的文件夹路径列表。
        """
        valid_folders = []

        # 遍历所有的 CSV 文件
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file)

            # 获取当前文件夹的名字作为基础路径
            base_folder = os.path.dirname(os.path.dirname(csv_file))

            # 检查每一行的 Bias 是否在范围内
            for _, row in df.iterrows():
                bias = row['Bias']
                if min_bias <= bias <= max_bias:
                    noise_type_params = str(row['Noise_Type_Params'])

                    if "gaussian_params" in noise_type_params:
                        folder_path = self.generate_folder_path(base_folder, "gaussian", noise_type_params)
                    elif "mosaic_params" in noise_type_params:
                        folder_path = self.generate_folder_path(base_folder, "mosaic", noise_type_params)
                    else:
                        folder_path = os.path.join(base_folder, "with_noise/raw")

                    valid_folders.append(folder_path)

        # 去重并按名称升序排列
        valid_folders = list(set(valid_folders))
        valid_folders.sort()  # 按名称升序排序
        return valid_folders

    def check_folders(self, folder_list):
        """
        检查文件夹是否存在，并确认每个文件夹下是否有 images 和 labels 子文件夹。
        
        Args:
            folder_list (List[str]): 需要检查的文件夹路径列表。
        """
        print("\nChecking folders:")
        for folder in folder_list:
            if os.path.exists(folder):
                images_folder = os.path.join(folder, 'images')
                labels_folder = os.path.join(folder, 'labels')
                if os.path.exists(images_folder) and os.path.exists(labels_folder):
                    print(f"[OK] {folder} contains 'images' and 'labels'.")
                else:
                    print(f"[WARN] {folder} is missing 'images' or 'labels'.")
            else:
                print(f"[ERROR] {folder} does not exist.")

# 使用示例
if __name__ == "__main__":
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/remix'
    
    plotter = NoiseFolderFinder(top_level_layer)
    
    # 查找所有包含 overall_evaluation_results.csv 的路径
    plotter.find_all_csv()
    
    # 设置 bias 的上下限
    min_bias = 235
    max_bias = 245

    # 获取所有符合条件的文件夹路径列表
    valid_folders = plotter.find_folders_by_bias_range(min_bias, max_bias)
    
    # 输出文件夹路径
    print("Folders with bias in the specified range (sorted):")
    for folder in valid_folders:
        print(folder)
    
    # 检查文件夹是否存在，并检查是否包含 images 和 labels 文件夹
    plotter.check_folders(valid_folders)
