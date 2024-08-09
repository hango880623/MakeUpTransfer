import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
import cv2

def mssim_loss(logit, target):
    """Compute multiscale structural similarity loss of the generative image."""
    loss = StructuralSimilarityIndexMeasure()
    return 1 - loss(logit, target)

if __name__ == '__main__':
    original_image = "./data/LABImage/p_3_55_5700.JPG"
    target_image = "./data/LABImage/p_1_55_5700.JPG"
    original = cv2.imread(original_image)
    target = cv2.imread(target_image)
    original = cv2.resize(original, (128, 128))
    target = cv2.resize(target, (128, 128))

    # Convert images to PyTorch tensors and add batch dimension
    original = torch.unsqueeze(torch.tensor(original).permute(2, 0, 1), 0).float()
    target = torch.unsqueeze(torch.tensor(target).permute(2, 0, 1), 0).float()

    print(mssim_loss(original, target))
