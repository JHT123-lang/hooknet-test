from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import geojson
from shapely.geometry import box, shape
from rtree import index
import albumentations as A
from torch.utils.data import Dataset
import openslide
import os

def extract_patches_and_masks(
    slide_path: str,
    geojson_path: str,
    output_dir: str,
    M: int = 256,
    r_C: int = 1,
    r_T: int = 4
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    output_dir = Path(output_dir) / Path(slide_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    patches_C_dir = output_dir / "patches_C"
    patches_T_dir = output_dir / "patches_T"
    masks_dir = output_dir / "masks"
    patches_C_dir.mkdir(exist_ok=True)
    patches_T_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    with open(geojson_path, "r") as f:
        geojson_data = geojson.load(f)

    idx = index.Index()
    features = geojson_data["features"]
    class_map = {"Background": 0, "AGG": 1, "FOL-I": 2, "FOL-II": 3}

    for i, feature in enumerate(features):
        geom = shape(feature["geometry"])
        idx.insert(i, geom.bounds)

    patches_C, patches_T, mask_patches, coordinates = [], [], [], []
    for y in range(0, height - M * r_C + 1, M * r_C):
        for x in range(0, width - M * r_C + 1, M * r_C):
            patch_C = slide.read_region((x, y), 1, (M * r_C, M * r_C))
            patch_C = np.array(patch_C.convert("RGB"))

            center_x = x + (M * r_C - M * r_T) // 2
            center_y = y + (M * r_C - M * r_T) // 2
            patch_T = slide.read_region((center_x, center_y), 0, (M * r_T, M * r_T))
            patch_T = np.array(patch_T.convert("RGB"))

            patch_box = box(center_x, center_y, center_x + M * r_T, center_y + M * r_T)
            mask_patch = np.zeros((M * r_T, M * r_T), dtype=np.uint8)
            img = Image.new("L", (M * r_T, M * r_T), 0)
            draw = ImageDraw.Draw(img)
            for i in idx.intersection(patch_box.bounds):
                geom = shape(features[i]["geometry"])
                if geom.intersects(patch_box):
                    clipped_geom = geom.intersection(patch_box)
                    classification = features[i]["properties"].get("classification", {"name": "Background"})
                    label_name = classification.get("name", "Background") if isinstance(classification, dict) else classification
                    label = class_map.get(label_name, 0)
                    if clipped_geom.geom_type == "Polygon":
                        coords = [(p[0] - center_x, p[1] - center_y) for p in clipped_geom.exterior.coords]
                        draw.polygon(coords, fill=label)
                    elif clipped_geom.geom_type == "MultiPolygon":
                        for poly in clipped_geom.geoms:
                            coords = [(p[0] - center_x, p[1] - center_y) for p in poly.exterior.coords]
                            draw.polygon(coords, fill=label)
            mask_patch = np.array(img)

            if np.any(mask_patch > 0):
                patch_id = f"{center_x}_{center_y}"
                np.save(patches_C_dir / f"patch_C_{patch_id}.npy", patch_C)
                np.save(patches_T_dir / f"patch_T_{patch_id}.npy", patch_T)
                np.save(masks_dir / f"mask_{patch_id}.npy", mask_patch)
                
                patches_C.append(patch_C)
                patches_T.append(patch_T)
                mask_patches.append(mask_patch)
                coordinates.append((center_x, center_y))

    return patches_C, patches_T, mask_patches, coordinates

def load_patches_and_masks(
    output_dir: str,
    slide_name: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    output_dir = Path(output_dir) / slide_name
    patches_C_dir = output_dir / "patches_C"
    patches_T_dir = output_dir / "patches_T"
    masks_dir = output_dir / "masks"

    patches_C, patches_T, mask_patches, coordinates = [], [], [], []
    if patches_C_dir.exists() and patches_T_dir.exists() and masks_dir.exists():
        for patch_file in patches_T_dir.glob("patch_T_*.npy"):
            patch_id = patch_file.stem.replace("patch_T_", "")
            coord_x, coord_y = map(int, patch_id.split("_"))
            
            patch_C = np.load(patches_C_dir / f"patch_C_{patch_id}.npy")
            patch_T = np.load(patch_file)
            mask_patch = np.load(masks_dir / f"mask_{patch_id}.npy")
            
            patches_C.append(patch_C)
            patches_T.append(patch_T)
            mask_patches.append(mask_patch)
            coordinates.append((coord_x, coord_y))
    
    return patches_C, patches_T, mask_patches, coordinates

class WSIDataset(Dataset):
    def __init__(self, patches_C, patches_T, mask_patches, transform=None):
        self.patches_C = patches_C
        self.patches_T = patches_T
        self.mask_patches = mask_patches
        self.transform = transform

    def __len__(self):
        return len(self.patches_T)

    def __getitem__(self, idx):
        context = self.patches_C[idx].astype(np.float32)
        target = self.patches_T[idx].astype(np.float32)
        mask = self.mask_patches[idx].astype(np.int64)

        if self.transform:
            aug_C = self.transform(image=context)
            aug_T = self.transform(image=target, mask=mask)
            context = aug_C["image"]
            target = aug_T["image"]
            mask = aug_T["mask"]

        return context, target, mask