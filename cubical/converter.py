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
        return [self.convert(image) for image in self.dataset]
    

class GrayscaleConverter(BaseConverter):
    def __init__(self, x1=0.299, x2=0.587, x3=0.114):
        """
        Initializes the GrayscaleConverter with weights for RGB channels.

        The weights (x1, x2, x3) are used to convert RGB images to grayscale images. 
        These weights represent the contribution of each RGB channel to the perceived brightness of a color.

        Args:
            x1 (float): Weight for the Red channel, between 0 and 1. Default is 0.299.
            x2 (float): Weight for the Green channel, between 0 and 1. Default is 0.587.
            x3 (float): Weight for the Blue channel, between 0 and 1. Default is 0.114.

        The default values (0.299, 0.587, 0.114) are derived from a common colorimetric conversion to grayscale 
        that takes into account human visual perception. Humans perceive green as the brightest color among the
        three primary colors (red, green, blue), which is why the weight for green is the highest. 
        This conversion is widely used in computer vision and image processing.
        """
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
    
    def set_weights(self, x1, x2, x3):
        """Set new weights for the RGB channels"""
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
    
    def convert(self, image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """The method to convert 3 channel images to 1 channel using the grayscale conversion formula."""

        # Check if the image is a PyTorch tensor
        if isinstance(image, torch.Tensor):
            # For PyTorch tensor
            R, G, B = image[0, :, :], image[1, :, :], image[2, :, :]
            grayscale = self.x1 * R + self.x2 * G + self.x3 * B
            return grayscale.unsqueeze(0)  # Adding a channel dimension for consistency

        # Check if the image is a numpy array
        elif isinstance(image, np.ndarray):
            # For numpy array
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            grayscale = self.x1 * R + self.x2 * G + self.x3 * B
            return grayscale[:, :, np.newaxis]  # Adding a channel dimension for consistency

        else:
            raise TypeError("Input type not supported. Only supports PyTorch tensor or numpy array.")

    def convert_dataset(self) -> List[Union[torch.Tensor, np.ndarray]]:
        """Perform the transform to total dataset."""
        
        # Check if the image is a PyTorch tensor
        if isinstance(self.dataset, torch.Tensor):
            # For PyTorch tensor
            R, G, B = self.dataset[:, 0, :, :], self.dataset[:, 1, :, :], self.dataset[:, 2, :, :]
            grayscale = self.x1 * R + self.x2 * G + self.x3 * B
            return grayscale.unsqueeze(1)  # Adding a channel dimension for consistency

        # Check if the image is a numpy array
        elif isinstance(self.dataset, np.ndarray):
            # For numpy array
            R, G, B = self.dataset[:, :, :, 0], self.dataset[:, :, :, 1], self.dataset[:, :, :, 2]
            grayscale = self.x1 * R + self.x2 * G + self.x3 * B
            return grayscale[:, :, np.newaxis]  # Adding a channel dimension for consistency

        else:
            raise TypeError("Input type not supported. Only supports PyTorch tensor or numpy array.")
