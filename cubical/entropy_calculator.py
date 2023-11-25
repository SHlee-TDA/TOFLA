"""
Module for topological data analysis vectorization.

This module provides the Vectorization class which is responsible for
transforming dataset samples using topological data analysis techniques,
and subsequently saving the transformed data.

Author: Seong-Heon Lee (Postech MINDS)
Date: 19. 10. 23
"""

import os
import time
import logging

import numpy as np
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations.vector_methods import Entropy
from tqdm import tqdm

from converter import BaseConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

persistence_0d = CubicalPersistence(
    homology_dimensions=0,
    input_type='vertices',
    homology_coeff_field=2,
    n_jobs=-1
)

persistence_1d = CubicalPersistence(
    homology_dimensions=1,
    input_type='vertices',
    homology_coeff_field=2,
    n_jobs=-1
)

vectorization = Entropy(mode='scalar')


class Vectorization:
    """
        A class to perform topological vectorization on datasets.

        This class provides functionalities to perform filtration, compute
        persistence diagrams, and vectorize the diagrams, and finally save
        the vectorized data.

        Attributes:
            dataset: A PyTorch Dataset.
            filtration: A filtration converter based on BaseConverter.
            persistence: Persistence diagram computation module.
            vectorization: Diagram vectorization module.
            vectors (np.ndarray, optional): Stores the vectorized data after transformation.
    """
    def __init__(self,
                dataset,
                filtration: BaseConverter,
                persistence,
                vectorization
                ):

        """
        Initialize the Vectorization instance.

        Args:
            dataset: A PyTorch Dataset.
            filtration (BaseConverter): Filtration method to convert samples.
            persistence: Persistence diagram computation module.
            vectorization: Diagram vectorization module.
        """
        self.dataset = dataset
        self.filtration = filtration
        if filtration.dataset is not dataset:
            self.filtration.dataset = self.dataset
        self.persistence = persistence
        self.vectorization = vectorization
        self.vectors = None

    def transform(self) -> np.ndarray:
        """
        Transform dataset samples using TDA techniques.

        This method performs the filtration, computes persistence diagrams,
        and vectorizes the diagrams.

        Returns:
            np.ndarray: The vectorized data.
        """
        start_time = time.time()

        # Step 1 : Convert sample to filtered image
        logger.info("Starting filtration...")
        filtered_dataset = self.filtration.convert_dataset()
        filtered_images = np.array([image[0].numpy() for image in filtered_dataset])
        
        # Step 2 : Compute Persistence Diagrams
        logger.info("Computing persistence diagrams...")
        persistence_diagrams = self.persistence.fit_transform(filtered_images)
        normalized_diagrams = [np.where(arr == np.inf, 1.0, arr) for arr in persistence_diagrams]
        
        # Step 3 : Vectorize Persistence Diagrams.
        logger.info("Vectorizing persistence diagrams")
        vectors = self.vectorization.fit_transform(normalized_diagrams)
        self.vectors = vectors
        
        end_time = time.time()
        logger.info(f"Total transformation time: {end_time - start_time} seconds")
        return vectors

