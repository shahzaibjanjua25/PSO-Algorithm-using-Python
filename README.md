# PSO Algorithm for Piece Packing

This repository contains a Python implementation of the Particle Swarm Optimization (PSO) algorithm for solving the piece packing problem. The goal of the algorithm is to find the best arrangement of given puzzle pieces in a two-dimensional space.

## Getting Started

To run the code, follow the steps below:

1. Clone the repository or download the code files.
2. Make sure you have Python 3 installed on your system.
3. Install the required dependencies by running the following command:

   ```
   pip install numpy matplotlib
   ```

4. Put the input data file (`jakobs.txt`) in the same directory as the code files. The input file should contain the description of the puzzle pieces and their vertices.

5. Execute the main script by running the following command:

   ```
   python piece_packing.py
   ```

6. The program will output the best position found by the PSO algorithm and its corresponding fitness value. It will also generate several plots for visualization purposes.

## Code Structure

The repository contains the following files:

- `piece_packing.py`: This is the main script that implements the PSO algorithm for piece packing.
- `jakobs.txt`: This is the input data file that contains the description of the puzzle pieces and their vertices.
- `README.md`: This file provides an overview of the code and instructions for running it.

## Dependencies

The code relies on the following Python libraries:

- `numpy`: Used for numerical operations and array manipulation.
- `matplotlib`: Used for generating plots and visualizations.

Make sure to install these dependencies using `pip` before running the code.

## Algorithm Overview

The PSO algorithm is a population-based optimization algorithm inspired by the behavior of bird flocks or fish schools. In this implementation, the algorithm tries to find the best arrangement of puzzle pieces by optimizing a fitness function that measures the overlap between the pieces.

The algorithm works as follows:

1. Load the puzzle piece data from the input file.
2. Initialize a population of particles, where each particle represents a potential solution.
3. Evaluate the fitness of each particle by calculating the overlap between its pieces.
4. Update the velocity and position of each particle based on its own best position and the global best position found so far.
5. Repeat steps 3 and 4 for a certain number of iterations or until a termination condition is met.
6. Output the best position found, which represents the optimal arrangement of puzzle pieces.

## Results

The program generates several plots to visualize the progress and results of the PSO algorithm. These plots include:

- **Swarm Fitness Plot**: Shows how the fitness of the global best position evolves over iterations.
- **Swarm Best Fitness Plot**: Shows how the fitness of the best particle in the swarm evolves over iterations.
- **Particle Positions in 2D**: Displays the positions of each particle in a 2D space for each dimension.
- **Particle Positions in 3D**: Displays the positions of each particle in a 3D space, where each dimension corresponds to a puzzle piece.

Additionally, the program also includes a random plot of accuracy and loss during iterations. This plot is not directly related to the PSO algorithm for piece packing but serves as an example of how to generate and display plots using `matplotlib`.

## Acknowledgments

The PSO algorithm implementation in this code is adapted from standard PSO algorithms and customized for the piece packing problem. The code can be further extended and optimized based on specific requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
