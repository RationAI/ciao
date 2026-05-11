import os
import random

from torchvision import datasets


target_dir = "CIAO_DATA_ROOT/pcam_sample"

os.makedirs(target_dir, exist_ok=True)

pcam_val = datasets.PCAM(root=target_dir, split="val", download=True)

indices = list(range(len(pcam_val)))
random.seed(42)
random.shuffle(indices)

count_0 = 0
count_1 = 0
total_wanted = 200
target_per_class = total_wanted // 2

for idx in indices:
    if count_0 >= target_per_class and count_1 >= target_per_class:
        break

    img, label = pcam_val[idx]

    if label == 0 and count_0 < target_per_class:
        img_path = os.path.join(target_dir, f"pcam_normal_{count_0:04d}.jpg")
        img.save(img_path)
        count_0 += 1
    elif label == 1 and count_1 < target_per_class:
        img_path = os.path.join(target_dir, f"pcam_tumor_{count_1:04d}.jpg")
        img.save(img_path)
        count_1 += 1

# we performed the search on the first 50 tumorous and first 50 non-tumorous images (sorted by names)
