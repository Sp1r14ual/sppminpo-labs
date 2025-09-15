import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def influence_functions(mu=0.0, sigma=1.0, alpha_list=[0.05, 0.10, 0.15]):
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 500)

    # Среднее
    IF_mean = x - mu

    # Медиана
    f_mu = norm.pdf(mu, loc=mu, scale=sigma)  # плотность в медиане
    IF_median = np.sign(x - mu) / (2 * f_mu)

    # Усечённые средние
    IF_trimmed = {}
    for alpha in alpha_list:
        lower = norm.ppf(alpha, loc=mu, scale=sigma)
        upper = norm.ppf(1 - alpha, loc=mu, scale=sigma)
        IF = np.zeros_like(x)
        mask = (x >= lower) & (x <= upper)
        IF[mask] = (x[mask] - mu) / (1 - 2*alpha)
        IF_trimmed[alpha] = IF

    # --- Построение графиков ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, IF_mean, label="Среднее")
    plt.plot(x, IF_median, label="Медиана")
    for alpha, IF in IF_trimmed.items():
        plt.plot(x, IF, label=f"Усечённое среднее α={alpha}")
    plt.legend()
    plt.title("Функции влияния оценок параметра сдвига")
    plt.xlabel("x")
    plt.ylabel("IF(x)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    influence_functions()