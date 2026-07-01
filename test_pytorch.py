# %% cell 1
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 1. Load the image using PIL
# Replace 'example_image.jpg' with your actual file path
img = Image.open("/tmp/flowers/flowers-102/jpg/image_07513.jpg")

# 2. Convert to a NumPy array
# Shape: (Height, Width, Channels) -> e.g., (1080, 1920, 3)
img_array = np.array(img)
np.set_printoptions(threshold=np.inf)
print(img_array)
print(f"Original shape: {img_array.shape}")

# 3. Transpose to PyTorch format (CHW)
# We swap axis 0 (Height) and 2 (Channels)
# New Shape: (Channels, Height, Width) -> e.g., (3, 1080, 1920)
img_transposed = img_array.transpose(2, 0, 1)
print(f"Transposed shape: {img_transposed.shape}")

# 4. Convert to Tensor and Normalize
# We use a transform or manual conversion to move to float32 and scale to [0, 1]
tensor_image = torch.from_numpy(img_transposed).float()
tensor_image /= 255.0

print(f"Final Tensor shape: {tensor_image.shape}")
print(f"Tensor Data Type: {tensor_image.dtype}")
print(f"Sample pixel value (first pixel): {tensor_image[0, 0, 0]}")


# %% cell 2
print("apple")
# %% cell 3
