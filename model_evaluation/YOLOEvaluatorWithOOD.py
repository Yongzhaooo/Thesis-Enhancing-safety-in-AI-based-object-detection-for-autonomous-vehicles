import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import shutil
from model_evaluation.YOLO_Evaluator import YOLOEvaluator
import cv2
from concurrent.futures import ThreadPoolExecutor
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
import shutil


class YOLOEvaluatorWithOOD(YOLOEvaluator):
    def __init__(self, model_path, train_mean_vector, yolo_processor, ood_threshold=240):
        super().__init__(model_path)
        self.train_mean_vector = train_mean_vector  # 用于 OOD 检测
        self.ood_threshold = ood_threshold  # OOD 阈值
        self.ood_excluded_images = []  # 记录 OOD 排除的图片路径
        self.yolo_processor = yolo_processor

    def extract_feature_vector(self, img_path):
        """提取图片的特征向量（基于 YOLO 模型或其他处理方式）"""
        # 直接传递图像路径，而不是 numpy 数组
        feature_vector = self.yolo_processor.calculate_feature_vector(img_path).cpu().numpy().flatten()
        return feature_vector
    
    def run_ood_detection(self, img_path):
        """执行 OOD 检测"""
        feature_vector = self.extract_feature_vector(img_path)
        relative_distance = np.linalg.norm(feature_vector - self.train_mean_vector)

        if relative_distance > self.ood_threshold:
            # 图片为 OOD
            return False  
        else:
            # 图片为 In-distribution
            return True  

    def evaluate_folder(self, img_folder, label_folder, ood_save_dir=None):
        """在评估文件夹中的所有图片前先进行 OOD 检测"""
        all_class_iou_values = defaultdict(list)
        img_files = list(Path(img_folder).glob('*.png'))
        label_files = [Path(label_folder) / (img_file.stem + '.txt') for img_file in img_files]

        # 创建 OOD 保存目录
        ood_save_dir = Path(ood_save_dir) if ood_save_dir else None
        if ood_save_dir:
            if ood_save_dir.exists():
                shutil.rmtree(ood_save_dir)
            ood_save_dir.mkdir(parents=True, exist_ok=True)

        non_ood_count = 0  # 记录通过 OOD 检测的图片数量

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for img_file, label_file in zip(img_files, label_files):
                if label_file.exists():
                    # OOD 检测
                    if self.run_ood_detection(img_file):
                        # OOD 检测通过，加入评估
                        non_ood_count += 1
                        future = executor.submit(self.evaluate_single_image, img_file, label_file)
                        futures.append(future)
                    else:
                        # OOD 检测未通过，将图片保存到 OOD 文件夹
                        if ood_save_dir:
                            shutil.copy2(img_file, ood_save_dir / img_file.name)
                        
            # 处理非 OOD 图片的评估结果
            for future in tqdm(futures, desc="Processing images"):
                class_iou_values, _ = future.result()
                for cls, iou in class_iou_values.items():
                    all_class_iou_values[cls].append(iou)

        # 如果没有通过 OOD 检测的图片，返回空结果
        if non_ood_count == 0:
            print("All images were classified as OOD. No evaluation was performed.")
            return {}, float('nan')  # 没有非 OOD 图片，返回空结果和 NaN

        # 计算每个类别和所有类别的平均 IoU
        mean_iou_per_class = {cls: np.mean(ious) if ious else 0.0 for cls, ious in all_class_iou_values.items()}
        mean_iou_across_all_classes = np.mean([iou for ious in all_class_iou_values.values() for iou in ious])

        return mean_iou_per_class, mean_iou_across_all_classes

    def performance(self, parent_folder, ood_save_dir=None):
        """递归遍历文件夹并先进行 OOD 检测再运行评估"""
        parent_path = Path(parent_folder)

        # 递归遍历所有子文件夹
        for root, dirs, files in os.walk(parent_path):
            img_folder = Path(root) / 'images'
            label_folder = Path(root) / 'correct_labels'
            
            # 仅处理同时包含 'images' 和 'correct_labels' 子文件夹的文件夹
            if img_folder.exists() and label_folder.exists():
                print(f"Evaluating folder: {root}")
                self.evaluate_folder(img_folder, label_folder, ood_save_dir=ood_save_dir)

        print("Evaluation completed.")
