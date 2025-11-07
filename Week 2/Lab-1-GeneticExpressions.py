import random

# Parameters
POP = 6
BITS = 5
CROSS = 0.7
MUT = 0.1
SEED = 42   # set None for random runs

if SEED is not None:
    random.seed(SEED)

def fit(s):
    """Fitness = square of decoded integer."""
    return int(s, 2) ** 2

def init_pop():
    return [''.join(random.choice('01') for _ in range(BITS)) for _ in range(POP)]

def select(p):
    """Tournament selection of 2, return best."""
    a, b = random.sample(p, 2)
    return a if fit(a) > fit(b) else b

def cross(a, b):
    """Single-point crossover."""
    if random.random() < CROSS:
        pt = random.randint(1, BITS - 1)
        return a[:pt] + b[pt:], b[:pt] + a[pt:]
    return a, b

def mutate(s):
    """Bit-flip mutation."""
    bits = list(s)
    for i in range(len(bits)):
        if random.random() < MUT:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)

def run(max_stall=5):
    """
    Run until best solution repeats (stagnation).
    max_stall = number of generations with no improvement before stopping.
    """
    pop = init_pop()
    best, best_fit = None, -1
    stall = 0
    gen = 0

    while stall < max_stall:
        gen += 1
        fits = [fit(x) for x in pop]
        gen_best = max(fits)
        gen_avg = sum(fits) / len(fits)
        best_idx = fits.index(gen_best)

        if gen_best > best_fit:
            best, best_fit = pop[best_idx], gen_best
            stall = 0  # reset if improvement
        else:
            stall += 1

        print(f"Gen {gen}: Best={best} (x={int(best,2)}), Fit={best_fit}, AvgFit={gen_avg:.2f}")

        nxt = []
        while len(nxt) < POP:
            p1, p2 = select(pop), select(pop)
            c1, c2 = cross(p1, p2)
            nxt.extend([mutate(c1), mutate(c2)])
        pop = nxt[:POP]

    print(f"\nStopped after {gen} generations (no improvement for {max_stall}).")
    print(f"Final Best: {best} (x={int(best,2)}), Fit={best_fit}")

if __name__ == "__main__":
    run()
