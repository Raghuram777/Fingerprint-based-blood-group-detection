import os
import shutil

source_dir = "fingerprint_data"
target_dir = "fingerprint_data_small"
samples_per_class = 50

# Create target_dir if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Loop through each class
for blood_group in os.listdir(source_dir):
    src_folder = os.path.join(source_dir, blood_group)
    dst_folder = os.path.join(target_dir, blood_group)

    if not os.path.isdir(src_folder):
        continue

    os.makedirs(dst_folder, exist_ok=True)

    # List and sort images
    images = sorted(os.listdir(src_folder))[:samples_per_class]
    
    for img_name in images:
        src_path = os.path.join(src_folder, img_name)
        dst_path = os.path.join(dst_folder, img_name)
        shutil.copyfile(src_path, dst_path)

print("✅ Done: Copied first 50 images from each class.")
