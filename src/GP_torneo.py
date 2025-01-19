import random
import math
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import warnings

# change the number for a different problem
data = np.load("../data/problem_5.npz")

X = data['x']  # Input variables
Y = data['y']  # Target values
print(X.shape)  # Print the shape of X
print(Y.shape)  # Print the shape of Y

# Number of input variables (determined by the shape of X)
NUM_VARIABLES = X.shape[0]

# Supported operators and functions for symbolic regression
OPERATORS = ["+", "*", "-", "/"]
FUNCTIONS = ["cos", "exp", "log", "sin", "sqrt", "square"]

# Define terminal variables (e.g., x1, x2, ...)
TERMINALS = ['x' + str(i) for i in range(1, NUM_VARIABLES + 1)]
# Map terminal variable names to their indices in X
VARIABLE_INDICES = {'x' + str(i): i - 1 for i in range(1, NUM_VARIABLES + 1)}

# Genetic programming configuration
POPULATION_SIZE = 700  # Number of individuals in the population
GENERATIONS = 500  # Number of generations for evolution
TOURNAMENT_SIZE = 20  # Number of individuals participating in a tournament
MUTATION_RATE = 0.2  # Probability of mutation
CROSSOVER_RATE = 0.8  # Probability of crossover
MAX_DEPTH = 6  # Maximum tree depth for individuals
MIN_DEPTH = 2  # Minimum tree depth for individuals
ELITE_SIZE = int(POPULATION_SIZE * 0.05)  # Number of elite individuals preserved each generation




# Classe per rappresentare un albero simbolico
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # The value of the node (operator, function, variable, or constant)
        self.left = left    # Left child (None if not applicable)
        self.right = right  # Right child (None if not applicable)

    def is_operator(self):
        # Check if the node is an operator
        return self.value in OPERATORS

    def is_function(self):
        # Check if the node is a function
        return self.value in FUNCTIONS

    def is_terminal(self):
        # Check if the node is a terminal (constant or variable)
        return not self.is_operator() and not self.is_function()
    
    def evaluate(self, variable_values):
        """
        Evaluate the expression represented by the tree.

        Args:
            variable_values (list): Input values for variables.

        Returns:
            float: The result of evaluating the expression.
        """
        try:
            if self.is_operator():
                # Evaluate left and right subtrees for binary operators
                left_val = self.left.evaluate(variable_values)
                right_val = self.right.evaluate(variable_values)

                # Perform the operation based on the operator
                if self.value == "+":
                    result = left_val + right_val
                elif self.value == "-":
                    result = left_val - right_val
                elif self.value == "*":
                    # Prevent overflow during multiplication
                    if abs(left_val) < 1e150 and abs(right_val) < 1e150:
                        result = left_val * right_val
                    else:
                        result = float('inf')
                elif self.value == "/":
                    # Handle division with safeguards
                    if abs(right_val) < 1e-20:  # Avoid division by very small values
                        result = 1e10
                    elif abs(left_val) > 1e50 or abs(right_val) > 1e50:  # Prevent overflow
                        result = 1e10
                    else:
                        result = left_val / right_val
                else:
                    raise ValueError(f"Unknown operator: {self.value}")

            elif self.is_function():
                # Evaluate the argument for unary functions
                arg = self.left.evaluate(variable_values)

                # Apply the function
                if self.value == "sin":
                    result = math.sin(arg)
                elif self.value == "cos":
                    result = math.cos(arg)
                elif self.value == "exp":
                    result = math.exp(arg) if arg < 700 else float('inf')  # Avoid overflow
                elif self.value == "log":
                    result = math.log(arg) if arg > 0 else -math.inf  # Log is undefined for non-positive values
                elif self.value == "sqrt":
                    result = math.sqrt(arg) if arg >= 0 else math.nan  # Sqrt is undefined for negative values
                elif self.value == "square":
                    result = arg ** 2 if abs(arg) < 1e75 else float('inf')  # Prevent overflow
                else:
                    raise ValueError(f"Unknown function: {self.value}")

            elif self.value in TERMINALS:
                # Retrieve the value of a variable terminal
                index = VARIABLE_INDICES[self.value]
                result = variable_values[index]
            else:
                # Treat the value as a numeric constant
                result = float(self.value)

            # Penalize invalid results (infinity or NaN)
            if math.isinf(result) or math.isnan(result):
                return 1e10
            return result

        except OverflowError:
            return 1e10  # Penalize overflow
        except Exception as e:
            print(f"Error during tree evaluation: {e}")
            return 1e10  # Penalize other runtime errors


    def __str__(self):
        # Convert the tree into a human-readable string representation
        if self.is_operator():
            return f"({self.left} {self.value} {self.right})"
        elif self.is_function():
            return f"{self.value}({self.left})"
        else:
            return str(self.value)

    def depth(self):
        # Calculate the depth of the tree
        if self.is_terminal():
            return 1
        elif self.is_operator():
            return 1 + max(self.left.depth(), self.right.depth())
        elif self.is_function():
            return 1 + self.left.depth()

    def copy(self):
        # Create a deep copy of the tree
        if self.is_terminal():
            return Node(self.value)
        elif self.is_operator():
            return Node(self.value, self.left.copy(), self.right.copy())
        elif self.is_function():
            return Node(self.value, self.left.copy())





# Generate the initial population using the Ramped Half-and-Half method
def generate_initial_population():
    population = []
    depth_range = range(MIN_DEPTH, MAX_DEPTH + 1)
    for depth in depth_range:
        for _ in range(POPULATION_SIZE // (len(depth_range) * 2)):
            tree = generate_tree(depth, method="full")
            population.append(tree)
            tree = generate_tree(depth, method="grow")
            population.append(tree)
    while len(population) < POPULATION_SIZE:
        tree = generate_tree(random.randint(MIN_DEPTH, MAX_DEPTH), method="grow")
        population.append(tree)
    return population



# Function to generate a random tree using "full" or "grow" methods
def generate_tree(max_depth, depth=0, method="grow"):
    if depth >= max_depth:
        # Terminale
        return Node(random.choice(TERMINALS + [str(random.uniform(-5, 5))]))
    else:
        if method == "full":
            if depth < max_depth - 1:
                node_type = random.choice(["operator", "function"])
            else:
                node_type = "terminal"
        else:  # method == "grow"
            node_type = random.choice(["operator", "function", "terminal"])

        if node_type == "operator":
            left = generate_tree(max_depth, depth + 1, method)
            right = generate_tree(max_depth, depth + 1, method)
            return Node(random.choice(OPERATORS), left, right)
        elif node_type == "function":
            child = generate_tree(max_depth, depth + 1, method)
            return Node(random.choice(FUNCTIONS), child)
        else:  # Terminale
            return Node(random.choice(TERMINALS + [str(random.uniform(-5, 5))])) 









# Improved fitness function (MSE with complexity penalty)
def fitness_function(tree, x_data, y_data, alpha=0.1):
    try:
        # Compute predictions using vectorized evaluation
        predictions = np.apply_along_axis(tree.evaluate, 0, x_data)  
        
        # Clip prediction values to avoid overflow
        predictions = np.clip(predictions, -1e50, 1e50)
        
        # Calculate the mean squared error (MSE) with overflow prevention
        errors = y_data - predictions
        errors = np.nan_to_num(errors, nan=1e10, posinf=1e10, neginf=1e10)  # Replace NaN or inf with large penalties
        squared_errors = np.square(errors)
        squared_errors = np.clip(squared_errors, 0, 1e50)  # Clip squared errors to prevent overflow
        mse = np.mean(squared_errors)  # Compute the MSE with increased numerical stability
        
        # Compute the complexity of the tree (e.g., number of nodes)
        complexity = count_nodes(tree)
        
        # Fitness function with complexity penalty (depending on the problem i change alpha)
        fitness = mse #+ alpha * complexity  
        
        return fitness
    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return float("inf")  # Penalize invalid trees



# Funzione di supporto per contare i nodi di un albero
def count_nodes(tree):
    if tree is None:
        return 0
    elif tree.is_terminal():
        return 1
    elif tree.is_operator():
        return 1 + count_nodes(tree.left) + count_nodes(tree.right)
    elif tree.is_function():
        return 1 + count_nodes(tree.left)

# Helper function to count the nodes in a tree
def tournament_selection(population, fitnesses):
    participants = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_index = min(participants, key=lambda idx: fitnesses[idx])
    return population[best_index].copy()

# Crossover for subtree
def subtree_crossover(parent1, parent2):
    def get_random_subtree(node):
        nodes = []
        def traverse(node):
            nodes.append(node)
            if node.left:
                traverse(node.left)
            if node.right:
                traverse(node.right)
        traverse(node)
        return random.choice(nodes)

    child1 = parent1.copy()
    child2 = parent2.copy()

    if random.random() < CROSSOVER_RATE:
        node1 = get_random_subtree(child1)
        node2 = get_random_subtree(child2)
        # Swap subtrees
        node1.value, node1.left, node1.right = node2.value, node2.left, node2.right
    return child1, child2

# Mutation by subtree
def subtree_mutation(tree):
    if random.random() < MUTATION_RATE:
        max_mutation_depth = random.randint(1, 3)
        mutation_subtree = generate_tree(max_mutation_depth)
        node = tree
        parents = []
        # find a random node
        while not node.is_terminal() and random.random() > 0.5:
            parents.append(node)
            if node.is_operator():
                node = random.choice([node.left, node.right])
            elif node.is_function():
                node = node.left
        # Replace the node with the new subtree
        node.value = mutation_subtree.value
        node.left = mutation_subtree.left
        node.right = mutation_subtree.right
    else:
        if tree.left:
            subtree_mutation(tree.left)
        if tree.right:
            subtree_mutation(tree.right)
    return tree

# Tree depth limitation
def limit_tree_depth(tree, max_depth):
    def trim(node, depth):
        if depth > max_depth:
            return Node(random.choice(TERMINALS + [str(random.uniform(-5, 5))])) 
        else:
            if node.left:
                node.left = trim(node.left, depth + 1)
            if node.right:
                node.right = trim(node.right, depth + 1)
            return node
    return trim(tree, 0)

# Genetic Programming Algorithm with Mechanisms to Avoid Fitness Plateau
def genetic_programming(x_data, y_data):
    """
    Genetic Programming algorithm with mechanisms to avoid fitness plateaus.

    Args:
        x_data (np.ndarray): Input data.
        y_data (np.ndarray): Output data.

    Returns:
        Node: The tree with the best fitness found.
    """
    population = generate_initial_population()
    best_tree = None
    best_fitness = float("inf")
    last_improvement = 0  # Track the latest generation with fitness improvement

    for generation in range(GENERATIONS):
       
        fitnesses = np.array(Parallel(n_jobs=-1)(delayed(fitness_function)(tree, x_data, y_data) for tree in population))
        min_fitness_index = np.argmin(fitnesses)
        min_fitness = fitnesses[min_fitness_index]

        # Update the best individual
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_tree = population[min_fitness_index].copy()
            last_improvement = generation
        
        if generation % 1 == 0:
            print(f"Generazione {generation + 1}, Miglior Fitness: {best_fitness}")
        
        if generation % 20 == 0:
            print(best_tree)
        
        # Elitism
        sorted_population = [tree for _, tree in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        elites = [tree.copy() for tree in sorted_population[:ELITE_SIZE]]

        # Selezione e riproduzione
        new_population = elites[:]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = subtree_crossover(parent1, parent2)
            child1 = subtree_mutation(child1)
            child2 = subtree_mutation(child2)
            child1 = limit_tree_depth(child1, MAX_DEPTH)
            child2 = limit_tree_depth(child2, MAX_DEPTH)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        # Adding random individuals to increase genetic diversity
        if generation % 20 == 0:  
            num_random_individuals = int(POPULATION_SIZE * 0.5)  
            for _ in range(num_random_individuals):
                new_population.append(generate_tree(random.randint(MIN_DEPTH, MAX_DEPTH), method="grow"))

        
        # Soft restart if no improvement for too many generations
        if generation - last_improvement > 50:  
            last_improvement = generation
            print("Soft restart") 
            for _ in range(int(POPULATION_SIZE * 0.5)):
                population[random.randint(0, POPULATION_SIZE - 1)] = generate_tree(random.randint(MIN_DEPTH, MAX_DEPTH), method="grow")
        
        population = new_population

    return best_tree

# Run GP algorithm
best_tree = genetic_programming(X, Y)


print("\nEspressione trovata:")
print(best_tree)




print("best fitness:",fitness_function(best_tree,X,Y))




# Funzione per raccogliere i terminali numerici dall'albero
def collect_numeric_terminals(tree):
    """Raccoglie i terminali numerici dell'albero."""
    numeric_terminals = []
    def traverse(node):
        if node.is_terminal() and node.value not in VARIABLE_INDICES:  # Se è un terminale numerico (non x1, x2, ecc.)
            try:
                val = float(node.value)
                numeric_terminals.append(node)
            except ValueError:
                pass
        if node.left:
            traverse(node.left)
        if node.right:
            traverse(node.right)
    traverse(tree)
    return numeric_terminals

# Funzione per aggiornare i terminali numerici con nuovi valori
def update_tree_with_new_values(tree, new_values):
    """Aggiorna i terminali numerici dell'albero con i nuovi valori."""
    terminals = collect_numeric_terminals(tree)
    for terminal, new_value in zip(terminals, new_values):
        terminal.value = str(new_value)  # Aggiorna il valore numerico come stringa
    return tree

# Funzione obiettivo per ottimizzare i terminali numerici
def mse_loss_for_tree(numeric_values, tree, x_data, y_data):
    """Calcola la MSE per l'albero con i terminali aggiornati."""
    updated_tree = update_tree_with_new_values(tree, numeric_values)
    predictions = np.apply_along_axis(updated_tree.evaluate, 0, x_data)
    errors = y_data - predictions
    mse = np.mean(np.square(errors))
    return mse

# Improved Hill Climbing algorithm with multi-restart and adaptive perturbation
def hill_climbing(tree, x_data, y_data, 
                  max_iter=1000, perturbation_scale=1.0, 
                  lower_bound=-10, upper_bound=10, 
                  restarts=5):
    """
    Optimize numeric terminals in the tree using a simple Hill Climbing algorithm.

    Args:
        tree (Node): The symbolic tree to optimize.
        x_data (np.ndarray): Input data.
        y_data (np.ndarray): Target data.
        max_iter (int): Maximum number of iterations per restart.
        perturbation_scale (float): Initial scale of random perturbations.
        lower_bound (float): Minimum value for numeric terminals.
        upper_bound (float): Maximum value for numeric terminals.
        restarts (int): Number of random restarts to escape local minima.

    Returns:
        Node: Optimized tree.
        float: Final MSE value.
        list: Optimized values for numeric terminals.
    """
    # Best solution across all restarts
    global_best_mse = float("inf")
    global_best_values = None
    global_best_tree = None

    for restart in range(restarts):
        print(f"Restart {restart + 1}/{restarts}")
        
        # Collect numeric terminals and initialize values
        numeric_terminals = collect_numeric_terminals(tree)
        current_values = np.array([float(terminal.value) for terminal in numeric_terminals])

        # Initialize current best state
        current_mse = mse_loss_for_tree(current_values, tree, x_data, y_data)
        best_values = current_values.copy()
        best_mse = current_mse

        for iteration in range(max_iter):
            # Propose a new solution by perturbing the current values
            new_values = current_values + np.random.uniform(-perturbation_scale, perturbation_scale, size=current_values.shape)
            new_values = np.clip(new_values, lower_bound, upper_bound)  # Ensure values stay within bounds

            # Evaluate the new solution
            new_mse = mse_loss_for_tree(new_values, tree, x_data, y_data)

            # Accept the new solution if it improves the objective
            if new_mse < current_mse:
                current_values = new_values
                current_mse = new_mse

                # Update the best solution if the new one is better
                if current_mse < best_mse:
                    best_values = current_values.copy()
                    best_mse = current_mse

            # Adjust perturbation scale adaptively
            perturbation_scale *= 0.99  # Gradual reduction of step size

            # Log progress periodically
            if iteration % 100 == 0 or iteration == max_iter - 1:
                print(f"Iteration {iteration + 1}: Current MSE = {current_mse:.6f}, Best MSE = {best_mse:.6f}")

        # Update global best if necessary
        if best_mse < global_best_mse:
            global_best_mse = best_mse
            global_best_values = best_values
            global_best_tree = tree.copy()
            update_tree_with_new_values(global_best_tree, global_best_values)

    print(f"\nGlobal best MSE after {restarts} restarts: {global_best_mse:.6f}")
    return global_best_tree, global_best_mse, global_best_values


# Execute improved Hill Climbing on numeric terminals in the best_tree
optimized_tree, final_mse, optimized_values = hill_climbing(
    best_tree, X, Y, max_iter=10000, perturbation_scale=1.0, 
    lower_bound=-10, upper_bound=10, restarts=5
)

# Print results
print("\nUpdated tree expression:")
print(optimized_tree)
print(f"Final MSE after optimization: {final_mse:.6f}")
print(f"Optimized values for numeric terminals: {optimized_values}")


# Calcola la nuova fitness per l'albero ottimizzato

print("fitness ottimizzata:",fitness_function(optimized_tree,X,Y))
