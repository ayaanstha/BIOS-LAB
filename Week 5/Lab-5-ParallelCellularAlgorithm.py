import numpy as np
from joblib import Parallel, delayed

# --- Objective Function (Rastrigin) ---
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


# --- Helper Function: Ensure bounds ---
def clip_to_bounds(x, lb, ub):
    return np.clip(x, lb, ub)


# --- Define PCA Algorithm ---
def parallel_cellular_algorithm(
    f, n_cells=100, dim=2, bounds=(-5.12, 5.12), max_iter=100, neighborhood_size=4
):
    lb, ub = bounds
    # Initialize cells randomly
    cells = np.random.uniform(lb, ub, (n_cells, dim))
    fitness = np.array([f(c) for c in cells])

    # Neighborhood topology (Von Neumann in 1D ring)
    def get_neighbors(idx):
        return [
            (idx - 1) % n_cells,  # left neighbor
            (idx + 1) % n_cells,  # right neighbor
        ]

    # Evolution
    for t in range(max_iter):
        def update_cell(i):
            neighbors = get_neighbors(i)
            local_best_idx = min(neighbors + [i], key=lambda j: fitness[j])
            local_best = cells[local_best_idx]

            # Mutation + local search
            new_solution = cells[i] + np.random.uniform(-1, 1, dim) * (
                local_best - cells[i]
            )
            new_solution = clip_to_bounds(new_solution, lb, ub)
            new_fit = f(new_solution)

            return new_solution if new_fit < fitness[i] else cells[i], min(
                new_fit, fitness[i]
            )

        results = Parallel(n_jobs=-1)(
            delayed(update_cell)(i) for i in range(n_cells)
        )

        cells, fitness = zip(*results)
        cells, fitness = np.array(cells), np.array(fitness)

        best_idx = np.argmin(fitness)
        if t % 10 == 0:
            print(f"Iteration {t:03d} | Best Fitness = {fitness[best_idx]:.5f}")

    best_idx = np.argmin(fitness)
    return cells[best_idx], fitness[best_idx]


# --- Run PCA on Rastrigin Function ---
best_solution, best_fitness = parallel_cellular_algorithm(
    rastrigin, n_cells=50, dim=2, max_iter=200
)

print("\nBest solution found:", best_solution)
print("Best fitness:", best_fitness)
