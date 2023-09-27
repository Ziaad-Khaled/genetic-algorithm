# genetic-algorithm

This project implements the basic features and operations of a Genetic Algorithm (GA) for optimizing objective functions. It includes operations such as selecting a mating pool from the current population, crossover, mutation of population samples, and evaluating objective and fitness functions. The algorithm is tested for two different objective functions, each with its own custom combination of operations.

## Objective Functions

### Objective Function 1 - Global Maximum Search

The first objective function is defined as:

𝑓(𝑥) = sin⁡(𝜋𝑥 / 256)

where 𝑥 is restricted to integers in the range [0, 255]. The goal is to find the global maximum of this function.

### Objective Function 2 - Global Minimum Search

The second objective function is defined as:

𝑓(𝑥, 𝑦) = (𝑥 − 3.14)² + (𝑦 − 2.72)² + sin(3𝑥 + 1.41) + sin(4𝑦 − 1.73)

where -5 ≤ 𝑥, 𝑦 ≤ 5. The objective is to find the global minimum of this function.

## Variable Representations

To accommodate different types of variables, the project supports the following representations:

1. Binary representation for the first objective function.
2. Real-valued representation for the second objective function.

## Genetic Algorithm Operations

### Crossover

The project provides different crossover options, including:

- 50% crossover (similar to the one implemented in PA01).
- One-point crossover (suitable for the first objective function).
- Two-points crossover (suitable for the first objective function).

### Mutation

Mutation operations are implemented, including:

- Mutation with repair function that accepts the dynamic range of variable values.
- Mutation with a variable-specific mutation probability.

## Visualization

- Visualizations for the generations of the GA are provided for both the first and second objective functions.

## Fitness Function

Appropriate fitness functions are used to evaluate the quality of solutions in the optimization process.
