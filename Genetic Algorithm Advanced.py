import numpy as np
import matplotlib.pyplot as plt


# Fitness function
def fitness_function(x1, x2):
    return 8 - (x1 + 0.0317) ** 2 + (x2) ** 2


# Initialize population
def init_population(npop, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (npop, 2))


# ----------------- OPERATOR DEFINITIONS -----------------

# SELECTION METHODS
def tournament_selection(pop, fitness_values, k):
    selected_indices = np.random.choice(len(pop), k, replace=False)
    selected_fitness = fitness_values[selected_indices]
    winner_index = selected_indices[np.argmax(selected_fitness)]
    return pop[winner_index]


def roulette_selection(pop, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = fitness_values / total_fitness
    cum_probs = np.cumsum(probabilities)
    r = np.random.rand()
    for i, cp in enumerate(cum_probs):
        if r <= cp:
            return pop[i]


# CROSSOVER METHODS
def arithmetic_crossover(parent1, parent2, pcross):
    if np.random.rand() < pcross:
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    return parent1, parent2


def single_point_crossover(parent1, parent2, pcross):
    if np.random.rand() < pcross:
        # For real-valued: swap variables instead of bits
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
        return child1, child2
    return parent1, parent2


# MUTATION METHODS
def gaussian_mutation(individual, sigma, pmut, lower_bound, upper_bound):
    for i in range(len(individual)):
        if np.random.rand() < pmut:
            individual[i] += np.random.normal(0, sigma)
            individual[i] = np.clip(individual[i], lower_bound, upper_bound)
    return individual


def bit_flip_mutation(individual, pmut, lower_bound, upper_bound):
    for i in range(len(individual)):
        if np.random.rand() < pmut:
            individual[i] = np.random.uniform(lower_bound, upper_bound)
    return individual


# ----------------- MAIN GA FUNCTION -----------------
def run_ga(npop, ngen, pcross, pmut, lower_bound, upper_bound, sigma, k,
           selection_method, crossover_method, mutation_method):
    pop = init_population(npop, lower_bound, upper_bound)
    best_hist, avg_hist = [], []

    for gen in range(ngen):
        fitness_values = np.array([fitness_function(x1, x2) for x1, x2 in pop])

        best_hist.append(np.max(fitness_values))
        avg_hist.append(np.mean(fitness_values))

        new_pop = []
        elite_indices = np.argsort(fitness_values)[-2:]  # Keep top 2 individuals
        new_pop.extend(pop[elite_indices])

        while len(new_pop) < npop:
            # Selection
            if selection_method == "tournament":
                parent1 = tournament_selection(pop, fitness_values, k)
                parent2 = tournament_selection(pop, fitness_values, k)
            else:
                parent1 = roulette_selection(pop, fitness_values)
                parent2 = roulette_selection(pop, fitness_values)

            # Crossover
            if crossover_method == "arithmetic":
                child1, child2 = arithmetic_crossover(parent1, parent2, pcross)
            else:
                child1, child2 = single_point_crossover(parent1, parent2, pcross)

            # Mutation
            if mutation_method == "gaussian":
                child1 = gaussian_mutation(child1, sigma, pmut, lower_bound, upper_bound)
                child2 = gaussian_mutation(child2, sigma, pmut, lower_bound, upper_bound)
            else:
                child1 = bit_flip_mutation(child1, pmut, lower_bound, upper_bound)
                child2 = bit_flip_mutation(child2, pmut, lower_bound, upper_bound)

            new_pop.extend([child1, child2])
            if len(new_pop) >= npop:
                break

        pop = np.array(new_pop[:npop])

    return best_hist, avg_hist


# ----------------- USER INTERFACE -----------------
def main():
    # Basic parameters
    npop = int(input("Enter population size: "))
    ngen = int(input("Enter number of generations: "))
    pcross = float(input("Enter crossover probability (0.0 to 1.0): "))
    pmut = float(input("Enter mutation probability (0.0 to 1.0): "))
    sigma = float(input("Enter standard deviation for Gaussian mutation: "))
    k = int(input("Enter tournament size (if using tournament selection): "))

    # Operator selection
    print("\nSelect operators:")
    selection_method = input("Selection method (tournament/roulette): ").lower()
    crossover_method = input("Crossover method (arithmetic/single_point): ").lower()
    mutation_method = input("Mutation method (gaussian/bit_flip): ").lower()

    # Validate inputs
    if selection_method not in ["tournament", "roulette"]:
        print("Invalid selection method. Using tournament as default.")
        selection_method = "tournament"

    if crossover_method not in ["arithmetic", "single_point"]:
        print("Invalid crossover method. Using arithmetic as default.")
        crossover_method = "arithmetic"

    if mutation_method not in ["gaussian", "bit_flip"]:
        print("Invalid mutation method. Using gaussian as default.")
        mutation_method = "gaussian"

    print("\nRunning Genetic Algorithm with:")
    print(f"  Population Size: {npop}")
    print(f"  Generations: {ngen}")
    print(f"  Crossover: {crossover_method} (P={pcross})")
    print(f"  Mutation: {mutation_method} (P={pmut})")
    print(f"  Selection: {selection_method}")

    # Run GA
    best, avg = run_ga(npop, ngen, pcross, pmut, -2, 2, sigma, k,
                       selection_method, crossover_method, mutation_method)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(best, label="Best Fitness", color='blue', linestyle='solid')
    plt.plot(avg, label="Average Fitness", color='red', linestyle='dashed')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"GA Performance ({selection_method}/{crossover_method}/{mutation_method})")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()



