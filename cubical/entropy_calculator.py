"""
This module provides functionality for calculating persistence entropy based on persistence diagrams 
obtained from topological data analysis (TDA). It includes a class for calculating the persistence entropy 
of datasets after applying filtration and computing persistence diagrams.
"""
from typing import List, Union

import numpy as np
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations.vector_methods import Entropy

from .converter import BaseConverter


def normalize_persistence_diagram(persistence_diagrams: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize the persistence diagrams by replacing infinity values with 1.0.

    Args:
        persistence_diagrams: List of numpy arrays representing persistence diagrams.

    Returns:
        List of normalized persistence diagrams.
    """
    return [np.where(persistence_pair == np.inf, 1.0, persistence_pair) for persistence_pair in persistence_diagrams]

def calculate_PD_for_dim(images: Union[np.ndarray, List[np.ndarray]], dimension: int) -> List[np.ndarray]:
    """
    Calculate the persistence diagrams for a given homology dimension.

    Args:
        images: List of images or a single numpy array representing images.
        dimension: The homology dimension to compute persistence diagrams for.

    Returns:
        List of persistence diagrams for the given dimension.
    """
    return normalize_persistence_diagram(
        CubicalPersistence(homology_dimensions=dimension, input_type='vertices', homology_coeff_field=2, n_jobs=-1).fit_transform(images)
    )


class EntropyCalculator:
    """
    A class to compute entropy from persistence diagrams for a given dataset.

    This class handles the process of filtration, computing persistence diagrams,
    and calculating the entropy of these diagrams.

    Attributes:
        filtration (BaseConverter): An instance of BaseConverter for data filtration.
        homology_dimension (Union[int, List[int]]): Homology dimensions to consider.
    """
    def __init__(self, filtration: BaseConverter, homology_dimension: Union[int, List[int]]) -> None:
        self.filtration = filtration
        self.dataset = self.filtration.dataset
        self.homology_dimension = homology_dimension
        self.entropy = None

    def compute(self) -> Union[dict, None]:
        """
        Perform the computation of entropy for the dataset.

        Returns:
            A dictionary of entropy values keyed by homology dimensions, 
            or None if computation is not possible.
        """
        filtered_dataset = self.filtration.convert_dataset()
        filtered_images = np.array([image.numpy() for image in filtered_dataset])

        persistence_diagrams = {}
        if isinstance(self.homology_dimension, int):
            persistence_diagrams[f'dim{self.homology_dimension}'] = calculate_PD_for_dim(filtered_images, self.homology_dimension)
        elif isinstance(self.homology_dimension, list):
            for dim in self.homology_dimension:
                persistence_diagrams[f'dim{dim}'] = calculate_PD_for_dim(filtered_images, dim)
        else:
            raise TypeError("Homology dimension must be an integer or a list of integers")

        self.entropy = {dim: Entropy('scalar').fit_transform(pd) for dim, pd in persistence_diagrams.items()}
        return self.entropy