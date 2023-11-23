# BaseConverter

## Overview

`BaseConverter` is an abstract base class that defines a general framework for image conversion classes in the library. It provides a template for converting images and a method to apply the conversion to a dataset.

## Usage
`BaseConverter` is designed to be subclassed by specific image conversion classes. It requires the implementation of the convert method, which defines how an individual image is converted.

## Class Documentation

`__init__(self, dataset)`
- Initializes the `BaseConverter` with a dataset.
- **Parameters**: 
  - `dataset`: The dataset to which the conversion will be applied.

`converter(self, image)`
- An abstract method that needs to be implemented in subclasses. It defines the conversion process for a single image.
- **Parameters**:
  - `image`: The image to be converted.

`convert_dataset(self)`
- Applies the `convert` method to each image in the dataset and returns the list of converted images.
- **Returns**: A list of converted images.


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