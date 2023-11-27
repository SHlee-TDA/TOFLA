"""
optimization.py

This script defines classes for optimization processes in the context of topological data analysis (TDA).
It includes an abstract base class, OptimizationProcess, and concrete implementations such as GridSearch and RandomSearch.
These classes are designed to find the optimal parameters that maximize entropy, derived from TDA calculations on image datasets.
"""
from abc import ABC, abstractmethod
import logging
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .converter import GrayscaleConverter
from .entropy_calculator import EntropyCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationProcess(ABC):
    """
    An abstract base class for optimization processes in topological data analysis.

    This class serves as a foundation for specific optimization strategies, like grid search or random search,
    aimed at finding the optimal parameters for image data transformations to maximize entropy.

    Attributes:
        dataset: The dataset on which the optimization process is performed.
        homology_dimension (int): The specific homology dimension used for calculating entropy in TDA.
        best_params (Optional[Tuple[float, float, float]]): The best found parameters for maximizing entropy.
        max_entropy (float): The maximum entropy value found during optimization.

    Methods:
        optimize(): Abstract method to be implemented by subclasses, defining the optimization strategy.
    """
    def __init__(self, dataset, homology_dimension: int):
        self.dataset = dataset
        self.homology_dimension = homology_dimension
        self.best_params: Optional[Tuple[float, float, float]] = None
        self.max_entropy: float = -float('inf')

    @abstractmethod
    def optimize(self) -> Tuple[Optional[Tuple[float, float, float]], float]:
        pass
    

class GridSearch(OptimizationProcess):
    """
    A grid search optimization process for finding the best parameters that maximize entropy.

    This class performs an exhaustive search over a specified parameter space to find the 
    combination of parameters that yields the highest entropy in the resulting data transformation.

    Attributes:
        dataset: The dataset to be processed.
        homology_dimension: The homology dimension used for entropy calculation.
        steps (int): The number of steps to divide the parameter space for each parameter.
    """

    def __init__(self, dataset, homology_dimension: int, steps: int = 10):
        """
        Initialize the grid search optimization process.

        Args:
            dataset: The dataset to be optimized.
            homology_dimension (int): The homology dimension to be used for entropy calculation.
            steps (int): The number of intervals to divide the range [0, 1] for each parameter.
        """
        super().__init__(dataset, homology_dimension)
        self.steps = steps
        self.entropy_data = None
        
    def optimize(self) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """
        Perform the grid search optimization.

        Iterates over a grid of parameter values, computing entropy for each combination and 
        identifying the combination that maximizes entropy.

        Returns:
            A tuple containing the best parameters and the maximum entropy achieved.
        """
        self.entropy_data = []
        for x1 in np.linspace(0, 1, self.steps):
            for x2 in np.linspace(0, 1, self.steps):
                for x3 in np.linspace(0, 1, self.steps):
                    converter = GrayscaleConverter(self.dataset, x1=x1, x2=x2, x3=x3)
                    entropy_calculator = EntropyCalculator(filtration=converter, homology_dimension=self.homology_dimension)
                    entropies = entropy_calculator.compute()
                    entropy_avg = sum(np.mean(values) for values in entropies.values())
                    
                    self.entropy_data.append(((x1, x2, x3), entropy_avg))

                    if entropy_avg > self.max_entropy:
                        self.max_entropy = entropy_avg
                        self.best_params = (x1, x2, x3)

                    logger.info(f"Params: ({x1:.4f}, {x2:.4f}, {x3:.4f}), Average Entropy: {entropy_avg:.4f}")

        return self.best_params, self.max_entropy

    def visualize_optimization(self):
        """
        Visualize the results of the grid search optimization.

        This method plots the entropy values across the parameter space using a 3D scatter plot.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        for (x1, x2, x3), entropy in self.entropy_data:
            ax.scatter(x1, x2, x3, c=entropy, cmap='viridis')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        plt.colorbar(ax.scatter)
        plt.show()


class RandomSearch(OptimizationProcess):
    def optimize(self):
        # Random search 로직 구현
        pass
