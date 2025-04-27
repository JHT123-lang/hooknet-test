import json
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
from models.HookNet import HookNet
from data.dataset import extract_patches_and_masks, load_patches_and_masks, WSIDataset
from utils.losses import DiceLoss
from train import train_model
from pathlib import Path
import logging
import os
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def main():
    logging.basicConfig(filename='hooknet2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting HookNet segmentation")

    with open("config.json", "r") as f:
        config = json.load(f)

    slides_dir = Path(config["slides_dir"])
    annotations_dir = Path(config["annotations_dir"])
    output_dir = Path(config["output_dir"])
    
    M = config["M"]
    r_C = config["r_C"]
    r_T = config["r_T"]
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    lambda_weight = config["lambda_weight"]
    depth = config["depth"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else config["device"])
    geojson_suffix = config.get("geojson_suffix", "")

    assert 2 ** depth * r_T >= r_C, f"分辨率约束不满足: 2^{depth} * {r_T} >= {r_C}"

    # 初始化模型、损失函数和优化器
    model = HookNet(n_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    slide_paths = list(slides_dir.glob("*.svs"))
    paired_files = []
    for slide_path in slide_paths:
        slide_name = slide_path.stem
        geojson_path = annotations_dir / f"{slide_name}{geojson_suffix}.geojson"
        if not geojson_path.exists():
            logging.warning(f"未找到 {slide_name}.svs 对应的 .geojson 文件，跳过")
            continue
        paired_files.append((str(slide_path), str(geojson_path), slide_name))

    if not paired_files:
        logging.error("未找到有效的切片-标注对")
        raise ValueError("未找到有效的切片-标注对")

    # 按WSI水平划分训练和验证集
    train_paired_files, val_paired_files = train_test_split(
        paired_files, test_size=0.2, random_state=42
    )
    logging.info(f"训练集WSI数量: {len(train_paired_files)}, 验证集WSI数量: {len(val_paired_files)}")

    if not train_paired_files:
        logging.error("训练集WSI为空")
        raise ValueError("训练集WSI为空")
    if not val_paired_files:
        logging.warning("验证集WSI为空，将无法进行验证")

    # 数据增强
    transform = A.Compose([
        A.Resize(M * r_T, M * r_T),
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # 逐个处理训练集WSI
    for slide_path, geojson_path, slide_name in train_paired_files:
        logging.info(f"处理训练集WSI: {Path(slide_path).name}")
        patches_dir = output_dir / slide_name / "patches_T"
        if patches_dir.exists() and any(patches_dir.glob("*.npy")):
            logging.info(f"加载 {slide_name} 的已有patches")
            patches_C, patches_T, mask_patches, coordinates = load_patches_and_masks(str(output_dir), slide_name)
        else:
            logging.info(f"提取 {slide_name} 的patches")
            patches_C, patches_T, mask_patches, coordinates = extract_patches_and_masks(
                slide_path, geojson_path, str(output_dir), M, r_C, r_T
            )

        if not patches_T:
            logging.warning(f"{slide_name} 未提取或加载到patches，跳过")
            continue

        # 创建训练数据集
        train_dataset = WSIDataset(patches_C, patches_T, mask_patches, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        logging.info(f"{slide_name} 训练数据集: {len(train_dataset)} patches")

        # 训练模型（无验证集，仅训练）
        logging.info(f"开始训练 {slide_name}")
        train_model(model, train_loader, None, criterion, dice_loss, optimizer, num_epochs, device, lambda_weight)

        # 清空内存
        del patches_C, patches_T, mask_patches, coordinates, train_dataset, train_loader
        torch.cuda.empty_cache()

    # 逐个处理验证集WSI（仅验证）
    for slide_path, geojson_path, slide_name in val_paired_files:
        logging.info(f"处理验证集WSI: {Path(slide_path).name}")
        patches_dir = output_dir / slide_name / "patches_T"
        if patches_dir.exists() and any(patches_dir.glob("*.npy")):
            logging.info(f"加载 {slide_name} 的已有patches")
            patches_C, patches_T, mask_patches, coordinates = load_patches_and_masks(str(output_dir), slide_name)
        else:
            logging.info(f"提取 {slide_name} 的patches")
            patches_C, patches_T, mask_patches, coordinates = extract_patches_and_masks(
                slide_path, geojson_path, str(output_dir), M, r_C, r_T
            )

        if not patches_T:
            logging.warning(f"{slide_name} 未提取或加载到patches，跳过")
            continue

        # 创建验证数据集
        val_dataset = WSIDataset(patches_C, patches_T, mask_patches, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        logging.info(f"{slide_name} 验证数据集: {len(val_dataset)} patches")

        # 验证模型（无训练集，仅验证）
        logging.info(f"开始验证 {slide_name}")
        train_model(model, None, val_loader, criterion, dice_loss, optimizer, num_epochs, device, lambda_weight)

        # 清空内存
        del patches_C, patches_T, mask_patches, coordinates, val_dataset, val_loader
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()