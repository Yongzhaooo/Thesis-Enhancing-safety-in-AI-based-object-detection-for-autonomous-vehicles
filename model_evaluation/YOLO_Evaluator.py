import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from shutil import copy2

class YOLOEvaluator:
    def __init__(self, model_path):
        self.model = torch.hub.load('/home/carla/yongzhao/thesis/thesis_yolo/', 'custom', path=model_path, source='local')

    def read_labels(self, txt_path, img_width, img_height):
        """读取标准结果（标签）并返回框的列表"""
        with open(txt_path, 'r') as file:
            boxes = []
            for line in file.readlines():
                data = line.strip().split()
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:])
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                boxes.append([x1, y1, x2, y2, class_id])
            return np.array(boxes)

    def create_mask(self, boxes, img_height, img_width):
        """基于给定的框生成掩码"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            mask[y1:y2+1, x1:x2+1] = 1
        return mask

    def calculate_iou(self, mask1, mask2):
        """计算两个掩码之间的IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union != 0 else 0
        return iou
    
    def match_detections_to_labels(self,label_boxes, detected_boxes, img_height, img_width):
        """使用最大IoU匹配策略计算每个类别的IoU"""
        class_iou_values = {}
        for cls in set(label_boxes[:, 4]):
            label_class_boxes = label_boxes[label_boxes[:, 4] == cls][:, :4]
            detected_class_boxes = detected_boxes[detected_boxes[:, 5] == cls][:, :4]

            if len(label_class_boxes) > 0 and len(detected_class_boxes) > 0:
                ious = np.zeros((len(label_class_boxes), len(detected_class_boxes)))
                for i, label_box in enumerate(label_class_boxes):
                    label_mask = self.create_mask([label_box], img_height, img_width)
                    for j, detected_box in enumerate(detected_class_boxes):
                        detected_mask = self.create_mask([detected_box], img_height, img_width)
                        ious[i, j] = self.calculate_iou(label_mask, detected_mask)
                best_iou = np.max(ious, axis=1)
                class_iou_values[cls] = np.mean(best_iou)
            else:
                class_iou_values[cls] = 0.0

        return class_iou_values


    def evaluate_single_image(self, img_path, txt_path, save_dir=None):
        """评估单张图片的IoU，并可选择保存结果"""
        try:
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]

            # 读取标准结果（标签）
            label_boxes = self.read_labels(txt_path, img_width, img_height)
            detected_boxes = self.model(img).xyxy[0].cpu().numpy()

            # classes_in_image = set(label_boxes[:, 4])  # 真实标签中的所有类
            # 使用最大IoU匹配策略
            class_iou_values = self.match_detections_to_labels(label_boxes, detected_boxes, img_height, img_width)

            # 计算所有类的平均IoU
            mean_iou = np.mean(list(class_iou_values.values())) if class_iou_values else 0.0

            # 保存结果
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                # 保存原始图像和标签文件
                copy2(img_path, save_dir / Path(img_path).name)
                copy2(txt_path, save_dir / Path(txt_path).name)

                # 保存原始图像、标准结果图像、检测结果图像等
                self.save_evaluation_results(img, label_boxes, detected_boxes, class_iou_values, mean_iou, save_dir)

            return class_iou_values, mean_iou

        except Exception as e:
            print(f"Error processing {img_path} and {txt_path}: {e}")
            return {}, 0.0

    def save_evaluation_results(self, img, label_boxes, detected_boxes, class_iou_values, mean_iou, save_dir):
        """保存评估结果，包括图片和IoU信息"""
        img_height, img_width = img.shape[:2]
        
        # 创建掩码
        label_mask = self.create_mask(label_boxes[:, :4], img_height, img_width)
        detected_mask = self.create_mask(detected_boxes[:, :4], img_height, img_width)
        
        # 保存带有检测框和掩码的图像
        self.save_image_with_boxes_and_masks(img, label_boxes, label_mask, save_dir / "original_with_standard_boxes_and_mask.png", "Original Image with Standard Boxes and Mask")
        self.save_image_with_boxes_and_masks(img, detected_boxes, detected_mask, save_dir / "detected_with_boxes_and_mask.png", "Detected Image with Boxes and Mask")
        
        # 保存IoU结果
        iou_result_path = save_dir / "iou_results.txt"
        with open(iou_result_path, 'w') as f:
            f.write('Class-wise IoU:\n')
            for cls, iou in class_iou_values.items():
                f.write(f'Class {cls:.1f}: IoU = {iou:.4f}\n')
            f.write(f'Mean IoU across all classes: {mean_iou:.4f}\n')

    def save_image_with_boxes_and_masks(self, image, boxes, mask, save_path, title):
        """在图像上绘制检测框和掩码，并保存图像"""
        img_with_boxes_and_masks = self.draw_masks_on_image(image, mask)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_with_boxes_and_masks, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(save_path), img_with_boxes_and_masks)
        print(f"{title} saved to {save_path}")

    def draw_masks_on_image(self, image, mask, color=(0, 0, 255), alpha=0.5):
        """在图像上绘制掩码"""
        mask_overlay = image.copy()
        mask_overlay[mask > 0] = color
        return cv2.addWeighted(mask_overlay, alpha, image, 1 - alpha, 0)

    def evaluate_folder(self, img_folder, label_folder, enable_single_evaluate=None):
        """评估文件夹中的所有图片"""
        all_class_iou_values = defaultdict(list)
        img_files = list(Path(img_folder).glob('*.png'))  # 假设图像文件格式为.png
        label_files = [Path(label_folder) / (img_file.stem + '.txt') for img_file in img_files]

        # 使用当前文件夹路径保存评估结果
        folder_results_save_dir = Path(img_folder).parent

        if enable_single_evaluate:
            single_save_dir = folder_results_save_dir
            single_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            single_save_dir = None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for img_file, label_file in zip(img_files, label_files):
                if label_file.exists():
                    if single_save_dir:
                        future = executor.submit(
                            self.evaluate_single_image,
                            img_file,
                            label_file,
                            single_save_dir / img_file.stem
                        )
                    else:
                        future = executor.submit(
                            self.evaluate_single_image,
                            img_file,
                            label_file
                        )
                    futures.append(future)

            for future in tqdm(futures, desc="Processing images"):
                class_iou_values, _ = future.result()
                for cls, iou in class_iou_values.items():
                    all_class_iou_values[cls].append(iou)

        mean_iou_per_class = {cls: np.mean(ious) if ious else 0.0 for cls, ious in all_class_iou_values.items()}
        mean_iou_across_all_classes = np.mean([iou for ious in all_class_iou_values.values() for iou in ious])

        # 保存结果到文件
        result_path = folder_results_save_dir / 'mean_iou.txt'
        with open(result_path, 'w') as f:
            f.write('Mean IoU per class:\n')
            for cls, mean_iou in sorted(mean_iou_per_class.items()):
                f.write(f'Class {cls:.1f}: Mean IoU = {mean_iou:.5f}\n')
            f.write(f'Mean IoU across all classes: {mean_iou_across_all_classes:.5f}\n')

        print(f"Mean IoU per class: {mean_iou_per_class}")
        print(f"Mean IoU across all classes: {mean_iou_across_all_classes}")

        return mean_iou_per_class, mean_iou_across_all_classes






    def run(self, parent_folder):
        """递归遍历文件夹并运行评估"""
        parent_path = Path(parent_folder)

        # 递归遍历所有子文件夹
        for root, dirs, files in os.walk(parent_path):
            img_folder = Path(root) / 'images'
            label_folder = Path(root) / 'labels'
            
            # 仅处理同时包含 'images' 和 'labels' 子文件夹的文件夹
            if img_folder.exists() and label_folder.exists():
                print(f"Evaluating folder: {root}")
                self.evaluate_folder(img_folder, label_folder)

        print("Evaluation completed.")
    
    


