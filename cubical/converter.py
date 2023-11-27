"""
This module defines classes for converting image datasets.
It includes a base class for image converters and a specific implementation 
for converting RGB images to grayscale.

BaseConverter is an abstract base class defining the template for image converters.
GrayscaleConverter extends BaseConverter to provide functionality for converting 
RGB images to grayscale images using specified weights for each color channel.
"""
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch


class BaseConverter(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def convert(self, image):
        """The method to convert 3 channel images to 1 channel"""
        pass

    def convert_dataset(self):
        """Perform the transform to total dataset."""
        pass    


class GrayscaleConverter(BaseConverter):
    def __init__(self, dataset, x1: float = 0.299, x2: float=0.587, x3: float = 0.114):
        """
        Initializes the GrayscaleConverter with weights for RGB channels.

        This converter uses the specified weights (x1, x2, x3) to transform RGB images into grayscale images. 
        The weights determine the contribution of each RGB channel to the perceived brightness of the resulting grayscale image. 
        The grayscale value (Y) of each pixel is computed as a weighted sum of the RGB values: Y = x1*R + x2*G + x3*B.

        Args:
            dataset: The dataset to be converted to grayscale. Each image in the dataset should be in RGB format.
            x1 (float): Weight for the Red channel, between 0 and 1. Default is 0.299.
            x2 (float): Weight for the Green channel, between 0 and 1. Default is 0.587.
            x3 (float): Weight for the Blue channel, between 0 and 1. Default is 0.114.

        The default values (0.299, 0.587, 0.114) are based on a standard colorimetric conversion to grayscale, 
        accounting for human visual perception. Green is perceived as the brightest of the primary colors (red, green, blue), 
        hence it has the highest weight in this conversion formula. This method is commonly used in computer vision and image processing.
        """
        super().__init__(dataset)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
    
    def set_weights(self, x1, x2, x3):
        """Update the weights for the RGB channels"""
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
    
    def convert(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert a 3-channel image to a 1-channel grayscale image.

        Args:
            image: The image to be converted, either a PyTorch tensor or a NumPy array.

        Returns:
            A grayscale image with a single channel.
        
        Raises:
            TypeError: If the input image type is not supported.
        """
        if isinstance(image, torch.Tensor):
            return self._convert_tensor(image)
        elif isinstance(image, np.ndarray):
            return self._convert_array(image)
        else:
            raise TypeError("Unsupported image type.")

    def convert_dataset(self) -> List[Union[torch.Tensor, np.ndarray]]:
        """
        Convert the entire dataset to grayscale using tensor operations.

        Returns:
            A tensor or array of grayscale images with shape 
            (batch size, 1, height, width) for torch.Tensor 
            or (batch size, height, width, 1) for np.ndarray.
        """
        
        # Check if the image is a PyTorch tensor
        if isinstance(self.dataset, torch.Tensor):
            # For PyTorch tensor
            R, G, B = self.dataset[:, 0, :, :], self.dataset[:, 1, :, :], self.dataset[:, 2, :, :]
            grayscale = (self.x1 * R + self.x2 * G + self.x3 * B) / 3
            return grayscale.unsqueeze(1)  # Adding a channel dimension for consistency

        # Check if the image is a numpy array
        elif isinstance(self.dataset, np.ndarray):
            # For numpy array
            R, G, B = self.dataset[:, :, :, 0], self.dataset[:, :, :, 1], self.dataset[:, :, :, 2]
            grayscale = (self.x1 * R + self.x2 * G + self.x3 * B) / 3
            return grayscale[:, :, np.newaxis]  # Adding a channel dimension for consistency

        else:
            raise TypeError("Input type not supported. Only supports PyTorch tensor or numpy array.")

    def _convert_tensor(self, image: torch.Tensor) -> torch.Tensor:
        R, G, B = image[0, :, :], image[1, :, :], image[2, :, :]
        return ((self.x1 * R + self.x2 * G + self.x3 * B) / 3).unsqueeze(0)

    def _convert_array(self, image: np.ndarray) -> np.ndarray:
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return ((self.x1 * R + self.x2 * G + self.x3 * B) / 3)[:, :, np.newaxis]