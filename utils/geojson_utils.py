# utils/geojson_utils.py
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def export_predictions(heatmap, features, output_dir, slide_name):
    output_dir = Path(output_dir) / slide_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar(label="AGG Probability")
    plt.savefig(output_dir / "heatmap.png")
    plt.close()

    geojson_data = {"type": "FeatureCollection", "features": features}
    with open(output_dir / "predictions.genjson", "w") as f:
        json.dump(geojson_data, f)

def generate_geojson_features(pred_mask, center_x, center_y, num_classes):
    features = []
    class_names = {1: "AGG", 2: "FOL-I", 3: "FOL-II"}
    for cls in range(1, num_classes):
        if np.any(pred_mask == cls):
            contours, _ = cv2.findContours(
                (pred_mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                coords = [[float(center_x + p[0][0]), float(center_y + p[0][1])] for p in contour]
                if len(coords) >= 3:
                    coords.append(coords[0])
                    feature = {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [coords]},
                        "properties": {"classification": class_names[cls]}
                    }
                    features.append(feature)
    return features