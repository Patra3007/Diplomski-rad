import cv2
import numpy as np
import os

prediction_dir = 'output/Fuji/RGB'
rgb_dir = 'datasets/Fuji/RGB'
output_dir = 'output/Fuji/blended_output'
os.makedirs(output_dir, exist_ok=True)

alpha = 0.35

for filename in os.listdir(prediction_dir):
    if filename.endswith('_pred.png'):
        base_name = filename.replace('_pred.png', '')
        rgb_filename = base_name + '.jpg'

        pred_path = os.path.join(prediction_dir, filename)
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        output_path = os.path.join(output_dir, base_name + '_blended.png')

        if not os.path.exists(rgb_path):
            print(f"[Warning] RGB image not found for {filename}, skipping.")
            continue

        rgb_image = cv2.imread(rgb_path)
        prediction_mask = cv2.imread(pred_path)

        if rgb_image is None:
            print(f"[Error] Failed to load RGB image: {rgb_path}")
            continue
        if prediction_mask is None:
            print(f"[Error] Failed to load prediction mask: {pred_path}")
            continue

        if rgb_image.shape != prediction_mask.shape:
            print(f"[Error] Size mismatch between RGB and prediction for {filename}, skipping.")
            continue

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        prediction_mask = cv2.cvtColor(prediction_mask, cv2.COLOR_BGR2RGB)

        apple_mask = np.all(prediction_mask == [255, 0, 0], axis=-1)

        if np.sum(apple_mask) == 0:
            print(f"[Note] No predicted apple pixels in {filename}, skipping blend.")
            blended = rgb_image  # Just copy original if nothing to blend
        else:
            overlay = rgb_image.copy()
            overlay[apple_mask] = [255, 0, 0]

            blended = rgb_image.copy()
            blended_pixels = cv2.addWeighted(
                rgb_image[apple_mask], 1 - alpha, overlay[apple_mask], alpha, 0
            )

            if blended_pixels is None:
                print(f"[Error] cv2.addWeighted failed for {filename}, skipping.")
                continue

            blended[apple_mask] = blended_pixels

        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, blended_bgr)
        print(f"[Saved] {output_path}")
