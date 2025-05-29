# Real-coded Genetic Algorithm (GA) Implementation

This project implements a flexible, interactive real-coded Genetic Algorithm in Python to optimize a nonlinear function with two continuous variables. The GA supports multiple operator choices for selection, crossover, and mutation, allowing dynamic experimentation with evolutionary strategies.

## Features

- Real-valued chromosome representation (2 variables per individual).
- Customizable population size and number of generations.
- Selection operators: Tournament selection and Roulette wheel selection.
- Crossover operators: Arithmetic crossover and Single-point variable swapping crossover.
- Mutation operators: Gaussian mutation and Bit-flip (random reset) mutation.
- Elitism: preserves the top 2 individuals each generation.
- Visualization of best and average fitness trends across generations using Matplotlib.
- User-friendly CLI prompts to set parameters and operator choices.

## How It Works

- Population Initialization: Generates a population of real-valued individuals within specified bounds (default -2 to 2).
- Fitness Evaluation: Applies a nonlinear fitness function defined as: F(x1, x2) = 8 - (x1 + 0.0317) ** 2 + (x2) ** 2 
- Selection: Chooses parents for reproduction using the selected method:
  - Tournament Selection: Picks the best out of k randomly chosen individuals.
  - Roulette Wheel Selection: Probability proportional to fitness.
- Crossover: Produces offspring by combining parents using:
  - Arithmetic Crossover: Weighted average of parent genes.
  - Single-Point Crossover: Swap individual variables between parents.
- Mutation: Introduces genetic diversity via:
  - Gaussian Mutation: Adds Gaussian noise to genes.
  - Bit-Flip Mutation: Randomly resets gene values within bounds.
- Elitism: Top 2 individuals automatically carried over to the next generation to retain best solutions.
- Iteration: The GA runs for a user-defined number of generations, tracking best and average fitness each generation.
- Visualization: Fitness trends plotted in real-time after completion

## Usage

- Run the script in a terminal or command prompt: python real_coded_ga.py
- Follow the prompts to input:
  - Population size (e.g., 50)
  - Number of generations (e.g., 100)
  - Crossover probability (0.0 to 1.0)
  - Mutation probability (0.0 to 1.0)
  - Standard deviation for Gaussian mutation (e.g., 0.1)
  - Tournament size (if using tournament selection)
  - Select operators for:
    - Selection (tournament or roulette)
    - Crossover (arithmetic or single_point)
    - Mutation (gaussian or bit_flip)
  - After execution, a plot will display the evolution of best and average fitness over generations.



