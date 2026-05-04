IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# resnet50.tiatoolbox-pcam uses no normalization (raw [0, 1] pixel values)
PCAM_MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
PCAM_STD: tuple[float, float, float] = (1.0, 1.0, 1.0)
