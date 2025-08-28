import os
import json
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

def convert_depth_npy_to_png(depth_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(depth_dir):
        if fname.endswith(".npy"):
            depth_path = os.path.join(depth_dir, fname)
            save_path = os.path.join(output_dir, fname.replace(".npy", ".png"))
            depth_array = np.load(depth_path)
            plt.imsave(save_path, depth_array, cmap='Greys')
            print(f"✅ Done: resized npy in depth and make pic")
            
def convert_json_to_mask(json_dir, output_dir, target_shape=(480, 640)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
        for entry in data.values():
            filename = entry['filename']
            mask = np.zeros(target_shape, dtype=np.uint8)
            for region in entry['regions'].values():
                shape = region['shape_attributes']
                x = np.array(shape['all_points_x']) * (target_shape[1] / 1300)
                y = np.array(shape['all_points_y']) * (target_shape[0] / 1300)
                pts = np.stack((x, y), axis=1).astype(np.int32)
                cv2.fillPoly(mask, [pts], 200)
            Image.fromarray(mask).save(os.path.join(output_dir, filename.replace(".jpg", ".png")))
            print(f"✅ Done: resized json and make pic")

def copy_and_resize_images(src_dir, dst_dir, target_shape=(480, 640)):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(src_dir, fname))
            resized = cv2.resize(img, (target_shape[1], target_shape[0]))
            cv2.imwrite(os.path.join(dst_dir, fname), resized)
            print(f"✅ Done: resized img")

def split_dataset(image_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=30):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    filenames = sorted([f for f in os.listdir(os.path.join(image_dir, 'RGB')) if f.endswith(('.jpg', '.png'))])
    random.seed(seed)
    random.shuffle(filenames)

    total = len(filenames)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = filenames[:train_end]
    val_files = filenames[train_end:val_end]
    test_files = filenames[val_end:]

    def save_split(file_list, filename):
        with open(os.path.join(output_dir, filename), "w") as f:
            for name in file_list:
                name_base = os.path.splitext(name)[0]
                rgb_path = f"RGB/{name_base}.jpg"
                label_path = f"Label/{name_base}.png"
                f.write(f"{rgb_path} {label_path}\n")

    save_split(train_files, "train4.txt")
    save_split(val_files, "val4.txt")
    save_split(test_files, "test4.txt")

    print(f"Saved: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")


def convert_png_to_jpg(folder_path, delete_original=False):
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    for png_file in png_files:
        png_path = os.path.join(folder_path, png_file)
        jpg_file = png_file.rsplit('.', 1)[0] + '.jpg'
        jpg_path = os.path.join(folder_path, jpg_file)

        # Otvori PNG i spremi kao JPG (konverzija RGB ako treba)
        with Image.open(png_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(jpg_path, "JPEG")

        if delete_original:
            os.remove(png_path)
            print(f"Converted and deleted: {png_file}")
        else:
            print(f"Converted: {png_file} -> {jpg_file}")

    print(f"\nDone. Converted {len(png_files)} images.")

def convert_mask_255_to_1(folder_path, output_folder=None, overwrite=False):
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    for file in files:
        input_path = os.path.join(folder_path, file)

        with Image.open(input_path) as img:
            mask = np.array(img)  # no conversion needed if already grayscale
            mask = np.where(mask == 255, 200, 0).astype(np.uint8)
            output_img = Image.fromarray(mask)

        if overwrite:
            output_img.save(input_path, format="PNG")
            print(f"Overwritten: {file}")
        else:
            output_path = os.path.join(output_folder, file)
            output_img.save(output_path, format="PNG")
            print(f"Saved to: {output_path}")

    print(f"\nDone. Processed {len(files)} images.")

# === USAGE ===
# Overwrite in place:
# convert_mask_255_to_1("path/to/your/folder", overwrite=True)

def prepare_fuji_dataset():
    base = "datasets/Fuji"
    os.makedirs(base, exist_ok=True)

    split_dataset("datasets/Fuji", "datasets/Fuji")

    print("✅ Fuji dataset prepared for DFormer!")

convert_json_to_mask("02-annotated_data_fuji/gt_json/test","anoted1")