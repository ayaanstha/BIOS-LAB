import numpy as np

def objective(x):
    return np.sum(x**2)  # Example (Sphere); replace with your function

def gwo(n=20, dim=2, lb=-10, ub=10, max_iter=200):
    wolves = np.random.uniform(lb, ub, (n, dim))
    fitness = np.array([objective(w) for w in wolves])

    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = np.inf, np.inf, np.inf

    for _ in range(max_iter):
        for i, wolf in enumerate(wolves):
            score = fitness[i]

            if score < alpha_score:
                alpha_score, alpha = score, wolf.copy()
            elif score < beta_score:
                beta_score, beta = score, wolf.copy()
            elif score < delta_score:
                delta_score, delta = score, wolf.copy()

        a = 2 - _ * (2/max_iter)

        for i in range(n):
            for leader, leader_pos in zip([alpha, beta, delta], [alpha, beta, delta]):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2*a*r1 - a
                C = 2*r2
                D = abs(C*leader_pos - wolves[i])
                X = leader_pos - A*D

                if leader is alpha:
                    X1 = X
                elif leader is beta:
                    X2 = X
                else:
                    X3 = X

            wolves[i] = (X1 + X2 + X3)/3

        wolves = np.clip(wolves, lb, ub)
        fitness = np.array([objective(w) for w in wolves])

    return alpha, alpha_score


best_gwo, fit_gwo = gwo()
print("Grey Wolf Best Solution:", best_gwo)
print("Fitness:", fit_gwo)
