import numpy as np

num_cities = 20

np.random.seed(42) # for reproducibility
city_coordinates = np.random.rand(num_cities, 2) * 100 # Coordinates between 0 and 100

# 1. Define ACO parameters
num_ants = 50
num_iterations = 200
pheromone_evaporation_rate = 0.5
pheromone_deposit_factor = 1.0

# 2. Calculate the distance matrix
distance_matrix = np.linalg.norm(city_coordinates[:, np.newaxis, :] - city_coordinates[np.newaxis, :, :], axis=2)

# Initialize pheromone trails
pheromone_trails = np.ones((num_cities, num_cities)) * 0.1

# Store the best tour found
best_tour = None
best_tour_length = float('inf')

for iteration in range(num_iterations):
    all_tours = []
    all_tour_lengths = []

    for ant in range(num_ants):
        # 3. Implement ant movement and 4. Tour construction
        current_city = np.random.randint(num_cities)
        tour = [current_city]
        visited_cities = {current_city}

        while len(tour) < num_cities:
            possible_next_cities = np.array([city for city in range(num_cities) if city not in visited_cities])
            if len(possible_next_cities) == 0:
                break

            # Calculate probabilities
            pheromone_values = pheromone_trails[current_city, possible_next_cities]
            heuristic_values = 1.0 / (distance_matrix[current_city, possible_next_cities] + 1e-9) # Add a small constant to avoid division by zero
            probabilities = (pheromone_values**1.0) * (heuristic_values**5.0) # Alpha and Beta parameters (typically between 1 and 5)
            probabilities /= probabilities.sum()

            # Select the next city
            next_city = np.random.choice(possible_next_cities, p=probabilities)
            tour.append(next_city)
            visited_cities.add(next_city)
            current_city = next_city

        # Complete the tour by returning to the starting city
        if len(tour) == num_cities:
            tour.append(tour[0])
            tour_length = sum(distance_matrix[tour[i], tour[i+1]] for i in range(num_cities))
            all_tours.append(tour)
            all_tour_lengths.append(tour_length)

            # Update the best tour found so far
            if tour_length < best_tour_length:
                best_tour_length = tour_length
                best_tour = tour

    # 5. Implement pheromone update rule
    pheromone_trails *= (1 - pheromone_evaporation_rate)

    # Deposit pheromone
    for tour, tour_length in zip(all_tours, all_tour_lengths):
        if tour_length > 0: # Avoid division by zero
            pheromone_deposit = pheromone_deposit_factor / tour_length
            for i in range(num_cities):
                pheromone_trails[tour[i], tour[i+1]] += pheromone_deposit
            pheromone_trails[tour[num_cities], tour[0]] += pheromone_deposit # Deposit on the edge returning to the start

    if (iteration + 1) % 50 == 0:
        print(f"Iteration {iteration + 1}/{num_iterations}, Best tour length: {best_tour_length:.2f}")

# 7. Keep track of the best tour found so far (already done within the loop)

print("\nACO algorithm completed.")
print("Best tour found:", best_tour)
print("Best tour length:", best_tour_length)
