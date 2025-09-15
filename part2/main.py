import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, skew, kurtosis

# === Класс распределения Хьюбера ===
import mpmath as mp
from mpmath import quad
from scipy.optimize import brentq
from scipy.stats import norm


class HuberDistribution:
    def __init__(self, v: float, mu: float = 0.0, sigma: float = 1.0):
        if not (0.0 < v < 1.0):
            raise ValueError("v must be in (0,1)")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.v = float(v)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def _solve_k(self):
        def eq(k):
            return 2.0 * norm.pdf(k) / k + 2.0 * (norm.cdf(k) - 1.0) - self.v / (1.0 - self.v)
        return float(brentq(eq, 1e-8, 20.0))

    def k(self): 
        return self._solve_k()

    def pdf(self, x: float) -> float:
        """PDF с учётом mu и sigma"""
        k = self.k()
        coeff = (1.0 - self.v) / (self.sigma * np.sqrt(2.0 * np.pi))
        z = (x - self.mu) / self.sigma
        if abs(z) <= k:
            return coeff * np.exp(-0.5 * z**2)
        else:
            return coeff * np.exp(0.5 * k**2 - k * abs(z))

    def P(self):
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)
        return float(quad(lambda t: coeff * mp.e**(-0.5 * t**2), [-k, k]))

    def _moment(self, n: int) -> float:
        """Моменты стандартизованного распределения (mu=0, sigma=1)."""
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)

        def central(t): return (t**n) * coeff * mp.e**(-0.5 * t**2)
        def tail(t): return (t**n) * coeff * mp.e**(0.5 * k**2 - k*t)

        I1 = quad(central, [0, k])
        I2 = quad(tail, [k, mp.inf])
        return float(2 * (I1 + I2))

    # === Теоретические характеристики ===
    def mean(self): 
        return self.mu  # симметрия распределения

    def variance(self): 
        return (self.sigma**2) * self._moment(2)

    def skewness(self): 
        return 0.0  # симметрия

    def excess_kurtosis(self):
        m2 = self._moment(2)
        m4 = self._moment(4)
        return m4 / m2**2 - 3

    # === Генерация выборки ===
    def sample(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        k, P = self.k(), self.P()
        samples, normal_buffer = [], []

        def next_normal():
            if normal_buffer:
                return normal_buffer.pop()
            u1, u2 = rng.uniform(), rng.uniform()
            r = np.sqrt(-2.0 * np.log(u1))
            z1, z2 = r*np.cos(2*np.pi*u2), r*np.sin(2*np.pi*u2)
            normal_buffer.append(z2)
            return z1

        for _ in range(size):
            r1 = rng.uniform()
            if r1 <= P:  # центр
                while True:
                    x1 = next_normal()
                    if abs(x1) <= k:
                        samples.append(x1)
                        break
            else:  # хвост
                r2 = rng.uniform()
                x2 = k - np.log(r2)/k
                samples.append(x2 if r1 <= (1+P)/2 else -x2)

        # масштабируем и сдвигаем
        return self.mu + self.sigma * np.array(samples)


# === Генерация данных ===
def pure_huber(N, v=0.3, mu=0.0, sigma=1.0, random_state=None):
    huber = HuberDistribution(v, mu, sigma)
    return huber.sample(size=N, random_state=random_state)


def contaminated_huber(N, v=0.3, mu=0.0, sigma=1.0, eps=0.2, noise_level=20.0, random_state=None):
    """10-30% наблюдений заменяем выбросами"""
    rng = np.random.default_rng(random_state)
    n_cont = int(N * eps)
    n_clean = N - n_cont
    clean = pure_huber(n_clean, v, mu, sigma, random_state=rng)
    contam = pure_huber(n_cont, v, mu, sigma * noise_level, random_state=rng)
    return np.concatenate([clean, contam])


# === Выборочные характеристики ===
def sample_characteristics(data):
    return {
        "mean": np.mean(data),
        "var": np.var(data, ddof=1),
        "skew": skew(data),
        "kurtosis": kurtosis(data, fisher=True),
        "median": np.median(data),
        "trim_mean_0.05": trim_mean(data, 0.05),
        "trim_mean_0.10": trim_mean(data, 0.10),
        "trim_mean_0.15": trim_mean(data, 0.15),
    }


# === Эксперимент ===
def experiment(N=500, v=0.3, eps=0.2, noise_level=20.0, random_state=42):
    rng = np.random.default_rng(random_state)
    huber = HuberDistribution(v, mu=0.0, sigma=1.0)

    datasets = {
        "pure": pure_huber(N, v=v, mu=0.0, sigma=1.0, random_state=rng),
        "contaminated": contaminated_huber(N, v=v, mu=0.0, sigma=1.0, noise_level=noise_level, eps=eps, random_state=rng),
    }

    results = {name: sample_characteristics(data) for name, data in datasets.items()}

    # Добавляем теоретические значения
    theoretical = {
        "mean": huber.mean(),
        "var": huber.variance(),
        "skew": huber.skewness(),
        "kurtosis": huber.excess_kurtosis(),
    }

    return results, datasets, theoretical


if __name__ == "__main__":
    results, datasets, theoretical = experiment(N=5000, v=0.3, eps=0.3, noise_level=65.0, random_state=1)

    print("\n=== THEORETICAL VALUES (Huber) ===")
    for k, v in theoretical.items():
        print(f"  {k:15s} = {v:.4f}")

    for dist_name, ests in results.items():
        print(f"\n{dist_name.upper()} (sample estimates):")
        for k, v in ests.items():
            print(f"  {k:15s} = {v:.4f}")

    # Визуализация
    plt.figure(figsize=(10, 6))
    for name, data in datasets.items():
        plt.hist(data, bins=50, density=True, alpha=0.5, label=name)
    plt.legend()
    plt.title("Гистограммы: Хьюбер (чистое и искажённое распределение)")
    plt.show()
