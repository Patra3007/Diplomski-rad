import json
import numpy as np
import cv2
import os
from collections import defaultdict

# --- SETTINGS ---
json_path = "jsons/val/via_region_data_modal.json"
output_dir = "output/Fuji/gt_masks"
target_size = (480, 640)  # height, width
original_size = (1300, 1300)  # change if needed

scale_y = target_size[0] / original_size[0]
scale_x = target_size[1] / original_size[1]

os.makedirs(output_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Process each image
for filename, file_data in data.items():
    print(f"Processing {filename}")
    
    mask = np.zeros(target_size, dtype=np.uint8)
    
    # --- Group all regions by apple_ID ---
    apple_regions = defaultdict(list)
    name = filename
    for region in file_data.get('regions', {}).values():
        shape_attr = region['shape_attributes']
        apple_id = region['region_attributes'].get('apple_ID')
        
        if not apple_id:
            continue  # skip if no apple_ID

        all_x = [int(x * scale_x) for x in shape_attr['all_points_x']]
        all_y = [int(y * scale_y) for y in shape_attr['all_points_y']]
        polygon = np.array([list(zip(all_x, all_y))], dtype=np.int32)
        apple_regions[apple_id].append(polygon)
    
    # --- Assign unique grayscale value per apple_ID ---
    for idx, (apple_id, polygons) in enumerate(apple_regions.items()):
        label_value = min((idx + 1), 255)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], color=label_value)
    
    # Save mask
    output_path = os.path.join(output_dir, filename.split('.')[0] + '.png')
    print(output_path)
    cv2.imwrite(output_path, mask)

print("✅ All masks generated correctly.")
