import random

# --- Objective Function (De Jong Sphere Function) ---
def evaluate(position):
    x, y = position
    return x**2 + y**2   # We aim to minimize this

# --- Hyperparameters ---
NUM_PARTICLES = 10
ITERATIONS = 50
INERTIA = 0.3       # Weight for previous velocity
COG_COEFF = 2.0     # Personal learning rate
SOC_COEFF = 2.0     # Social learning rate

# --- Particle Initialization ---
positions = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(NUM_PARTICLES)]
vels = [[0.0, 0.0] for _ in range(NUM_PARTICLES)]

# --- Personal and Global Bests ---
personal_best_positions = [pos[:] for pos in positions]
personal_best_scores = [evaluate(pos) for pos in positions]

best_particle_idx = personal_best_scores.index(min(personal_best_scores))
global_best_position = personal_best_positions[best_particle_idx][:]
global_best_score = personal_best_scores[best_particle_idx]

# --- PSO Iterations ---
for step in range(ITERATIONS):
    for i in range(NUM_PARTICLES):
        r1, r2 = random.random(), random.random()

        # --- Velocity Update ---
        for d in range(2):  # For each dimension
            inertia_term = INERTIA * vels[i][d]
            cognitive_term = COG_COEFF * r1 * (personal_best_positions[i][d] - positions[i][d])
            social_term = SOC_COEFF * r2 * (global_best_position[d] - positions[i][d])
            vels[i][d] = inertia_term + cognitive_term + social_term

        # --- Position Update ---
        positions[i][0] += vels[i][0]
        positions[i][1] += vels[i][1]

        # --- Evaluate New Fitness ---
        score = evaluate(positions[i])

        # --- Update Personal & Global Bests ---
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i][:]

            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i][:]

    print(f"Iteration {step + 1:02d}/{ITERATIONS} → Best Score: {global_best_score:.6f} at {global_best_position}")

# --- Results ---
print("\n✅ Optimization Complete:")
print(f"→ Best Position Found: {global_best_position}")
print(f"→ Minimum Value: {global_best_score:.6f}")
