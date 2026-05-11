import os

from datasets import load_dataset


HF_TOKEN = ""

target_dir = "images"
images_dir = os.path.join(target_dir, "images")
masks_dir = os.path.join(target_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Načteme dataset
dataset = load_dataset(
    "braceletboy/imagenet-s", split="validation", streaming=True, token=HF_TOKEN
)

dataset = dataset.shuffle(seed=42, buffer_size=500)

for i, item in enumerate(dataset):
    if i >= 500:
        break

    img = item["image"]
    mask = item["mask"]

    img_path = os.path.join(images_dir, f"imagenets_{i:04d}.jpg")
    mask_path = os.path.join(masks_dir, f"imagenets_{i:04d}.png")

    img.convert("RGB").save(img_path)
    mask.convert("RGB").save(mask_path)

    if (i + 1) % 10 == 0:
        print(f"Saved {i + 1} / 500...")


# we performed the grid search and stability on the first 30 images (sorted by names),
# the comparison was performed on the next 100 images (0030 -> 0129)
