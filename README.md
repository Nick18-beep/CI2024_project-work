# Symbolic Regression with Genetic Programming

## Project Overview

This project involves the development of a Python program that uses a **Genetic Programming (GP)** algorithm to tackle symbolic regression problems. The primary objective is to discover interpretable mathematical functions that accurately describe the provided data while balancing **accuracy** and **complexity**.

### Key Features

- **Function Representation**: 
  Functions are represented as symbolic trees, where nodes correspond to operators, mathematical functions, or variables.
  
- **Initialization**: 
  The initial population is generated using the **Ramped Half-and-Half** method, ensuring diversity with varying depths and structures.

- **Evaluation**: 
  Trees are evaluated based on the **Mean Squared Error (MSE)** against the data, with a complexity penalty to encourage simplicity.

- **Evolutionary Operators**:
  - **Selection**: Tournament-based selection of the best-performing trees.
  - **Crossover**: Subtree swapping between parent trees to create new solutions.
  - **Mutation**: Random modification of subtrees to introduce variability.
  - **Elitism**: Preservation of top-performing individuals for continuity.
  - **Soft Restart**: Injection of new random individuals when stagnation is detected.

- **Diversity Maintenance**:
  Every 20 generations, 50% of the population is replaced with new random individuals to enhance genetic diversity.

---

## Evolution Process

1. **Elitism**: Top-performing individuals are directly copied into the next generation.
2. **Crossover**: Subtrees are swapped between parents with a predefined crossover rate, introducing new structures.
3. **Mutation**: Random subtrees within individuals are replaced with new subtrees while maintaining depth constraints.
4. **Diversity Boost**: Periodic introduction of random individuals to avoid premature convergence.

This balance of **exploration** (mutation and diversity) and **exploitation** (elitism and selection) drives the algorithm toward optimal solutions.

---

## Parameter Selection and Challenges

- **Challenging Problems**: Problems 2, 3, 7, and 8 required significant adjustments:
  - Increased population size and number of generations for better search space exploration.
  - Expanded numeric terminal range for problem 2 to improve solution quality.

- **Function Set**: 
  Limited to basic functions: `["cos", "exp", "log", "sin", "sqrt", "square"]`. These functions offer flexibility without unnecessarily enlarging the search space.

---

## Optimization Techniques

### Code Optimization
- **Fitness Evaluation**: 
  Replaced `numpy` with Python's `math` library for tree evaluation, as `math` is more efficient for recursive and sequential operations.

### Numerical Terminal Optimization
- **Method**: 
  A **Hill Climbing** algorithm with multiple restarts optimizes numerical terminals (constants or coefficients) in symbolic trees.
- **Process**:
  - Random perturbations of terminal values within a defined range.
  - Minimization of **MSE** by updating tree predictions with optimized values.
  - Multiple restarts to avoid local minima.

This fine-tuning process significantly improves the accuracy of symbolic trees, resulting in better-fitting models.

---

## Key Insights

1. Expanding the numeric terminal range allows the algorithm to explore a broader solution space (in particular for problem 2, in the other case the range was the same -10 to 10).
2. Limiting the function set while allowing deeper trees simplifies optimization without sacrificing expressiveness.
3. Parallelized fitness evaluation and the use of efficient libraries can greatly enhance computational performance.
4. Introducing random individuals periodically ensures the algorithm avoids stagnation and maintains innovation.

---
