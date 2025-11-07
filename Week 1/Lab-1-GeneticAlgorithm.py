import random, math

def fitness(x):
    return x * math.sin(10 * math.pi * x) + 2

def init_pop(size):
    return [random.uniform(-1, 2) for _ in range(size)]

def select(pop, fits):
    # shift fitnesses if any are negative (roulette requires non-negative)
    min_fit = min(fits)
    if min_fit < 0:
        fits = [f - min_fit + 1e-6 for f in fits]

    total = sum(fits)
    r = random.uniform(0, total)
    cum = 0
    for p, f in zip(pop, fits):
        cum += f
        if cum >= r:
            return p
    return random.choice(pop)

def crossover(a, b, rate):
    if random.random() < rate:
        alpha = random.random()
        return alpha * a + (1 - alpha) * b, alpha * b + (1 - alpha) * a
    return a, b

def mutate(x, rate, sigma=0.1):
    if random.random() < rate:
        return min(2, max(-1, x + random.gauss(0, sigma)))
    return x

def run():
    POP = 6
    CROSS, MUT = 0.8, 0.05
    pop = init_pop(POP)
    best, best_fit = None, float("-inf")
    stall, max_stall = 0, 5
    gen = 0

    while stall < max_stall:  # stop when no improvement
        gen += 1
        fits = [fitness(x) for x in pop]
        gen_best_fit = max(fits)
        gen_best = pop[fits.index(gen_best_fit)]

        if gen_best_fit > best_fit:
            best, best_fit = gen_best, gen_best_fit
            stall = 0
        else:
            stall += 1

        avg_fit = sum(fits) / len(fits)
        print(f"Gen {gen}: Best Fitness = {best_fit:.4f}, x = {best:.4f}, AvgFit = {avg_fit:.4f}")

        new_pop = []
        while len(new_pop) < POP:
            p1, p2 = select(pop, fits), select(pop, fits)
            c1, c2 = crossover(p1, p2, CROSS)
            new_pop += [mutate(c1, MUT), mutate(c2, MUT)]
        pop = new_pop[:POP]

    print(f"\nStopped after {gen} generations (no improvement for {max_stall}).")
    print(f"Best solution found: x = {best:.4f}, f(x) = {best_fit:.4f}")

if __name__ == "__main__":
    run()
