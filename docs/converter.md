# GrayscaleConverter

## Overview

The `GrayscaleConverter` class provides functionality to convert RGB images to grayscale images. This conversion applies weights to each RGB channel. The class supports images in both `PyTorch` tensor and `NumPy` array formats.

## Usage
```
from converter import GrayscaleConverter

# Load an example image
# image = loaded image

# Set weights (optional)
x1, x2, x3 = 0.299, 0.587, 0.114

# Create an instance of GrayscaleConverter
converter = GrayscaleConverter(x1, x2, x3)

# Convert the image
grayscale_image = converter.convert(image)
```

## Class Documentation

### GrayScaleConverter
`__init__(self, x1=0.299, x2=0.587, x3=0.114)`
    - Initializes weights used for converting RGB images to grayscale.
    -  **Parameters**:
      - `x1` (float): Weight for the Red channel. Default is 0.299.
      - `x2` (float): Weight for the Green channel. Default is 0.587.
      - `x3` (float): Weight for the Blue channel. Default is 0.114.

`convert(self, image)`
  - Converts a given RGB image to grayscale.
  - **Parameters**:
    - `image` (Union[torch.Tensor, np.ndarray]): The image to be converted.
  - **Returns**: The grayscale image.
`convert_dataset(self)`
    - Converts all images in the dataset stored in the class to grayscale.
    - **Returns**: A list of converted images.