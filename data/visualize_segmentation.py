import cv2
import mediapipe as mp
import numpy as np
import glob
import os

# Path to a few sample images
sample_images_dir = os.path.join(os.path.dirname(__file__), "images for phrases", "ALL")
sample_images = []
for ext in ("*.jpg", "*.png", "*.jpeg"):
    sample_images.extend(glob.glob(os.path.join(sample_images_dir, ext)))
# Take first 5 images for visualization
sample_images = sample_images[:5]

# Path to backgrounds
backgrounds_dir = os.path.join(os.path.dirname(__file__), "..", "backgrounds")
backgrounds = []
for ext in ("*.jpg", "*.png", "*.jpeg"):
    backgrounds.extend(glob.glob(os.path.join(backgrounds_dir, ext)))

mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

for idx, img_path in enumerate(sample_images):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_results = segmentor.process(img_rgb)
    if seg_results.segmentation_mask is not None:
        mask = seg_results.segmentation_mask
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"seg_mask_{idx}.png", mask_vis)
        print(f"Saved mask for {img_path} as seg_mask_{idx}.png")
        # Visualize blended image
        if backgrounds:
            bg_img_path = np.random.choice(backgrounds)
            bg_img = cv2.imread(bg_img_path)
            if bg_img is not None:
                bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
                mask3 = np.stack([mask]*3, axis=-1)
                mask3 = (mask3 > 0.1).astype(np.uint8)
                blended = (img * mask3 + bg_img * (1 - mask3)).astype(np.uint8)
                cv2.imwrite(f"blended_{idx}.png", blended)
                print(f"Saved blended image for {img_path} as blended_{idx}.png")
segmentor.close()
print("Visualization complete. Check seg_mask_*.png and blended_*.png in your folder.")
