# TOFLA (Topologically Optimized Filtration Learning Algorithm)
**TOFLA** is an innovative algorithm designed to optimize grayscale conversion mappings for RGB image data from a topological perspective, enhancing Topological Data Analysis (TDA) performance. This package provides tools to find an optimal grayscale converter that maximizes topological feature in RGB images.

## Algorithm Overview
TOFLA follows a structured approach to optimize the conversion of RGB images to grayscale for effective TDA:

1. **Linear Grayscale Conversion**: Converts RGB images to grayscale using a linear model: 
        $ Y = x_1 \times R + x_2 \times G + x_3 \times B $
    ,where $x_1, x_2$ and $x_3$ are parameters to be optimized. 
2. **Filtration**: Applies the grayscale conversion to create filtered images, leading to the construction of cubical complexes.
3. **Persistence Diagram Computation**: Calculates persistence diagrams from these cubical complexes to encode topological features.
4. **Persistence Entropy Calculation**: Computes the persistence entropy for each diagram, a measure of the complexity and richness of the topological features.
5. **Optimization**: Averages the persistence entropy across the dataset and optimizes $x_1, x_2$ and $x_3$ to maximize this average value.

## Key Concepts
- **Grayscale Converter**: A mechanism to transform RGB images to grayscale. In TOFLA, this conversion is optimized to retain significant topological features.
- **Cubical Complex**: A topological space constructed from the filtered grayscale images, used in TDA to study the shape and features of the data.
- **Persistent Homology**: A method in TDA to analyze and summarize the shapes in the data. It helps in understanding the structure and connectivity of the data at multiple scales.
- **Persistence Entropy**: A measure of the complexity of the topological features in the data. Higher entropy implies richer topological information.

## Usage Example
TBA

## Requirements

- `python >= 3.10`
- `pytorch >= 2.10`
- `gudhi >= 3.8.0`
