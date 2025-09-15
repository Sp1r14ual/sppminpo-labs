import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from scipy.optimize import minimize, brentq


# === Класс распределения Хьюбера ===
import mpmath as mp
from mpmath import findroot, quad
from scipy.stats import norm


class HuberDistribution:
    def __init__(self, v: float):
        if not (0.0 < v < 1.0):
            raise ValueError("v must be in (0,1)")
        self.v = float(v)

    def _solve_k(self):
        def eq(k):
            return 2.0 * norm.pdf(k) / k + 2.0 * (norm.cdf(k) - 1.0) - self.v / (1.0 - self.v)
        return float(brentq(eq, 1e-8, 20.0))

    def k(self): return self._solve_k()

    def pdf(self, x: float) -> float:
        k = self.k()
        coeff = (1.0 - self.v) / np.sqrt(2.0 * np.pi)
        ax = abs(x)
        if ax <= k:
            return coeff * np.exp(-0.5 * x**2)
        else:
            return coeff * np.exp(0.5 * k**2 - k * ax)

    def P(self):
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)
        return float(quad(lambda t: coeff * mp.e**(-0.5 * t**2), [-k, k]))

    def _moment(self, n: int) -> float:
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)

        def central(t): return (t**n) * coeff * mp.e**(-0.5 * t**2)
        def tail(t): return (t**n) * coeff * mp.e**(0.5 * k**2 - k*t)

        I1 = quad(central, [0, k])
        I2 = quad(tail, [k, mp.inf])
        return float(2 * (I1 + I2))

    def variance(self): return self._moment(2)
    def excess_kurtosis(self):
        m2 = self._moment(2)
        m4 = self._moment(4)
        return m4 / m2**2 - 3

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

        return np.array(samples)


# === Оценки сдвига ===
def mle_estimate(data):
    res = minimize(lambda mu: np.sum((data - mu) ** 2), x0=np.mean(data))
    return res.x[0]


def radical_estimate(data, c=1.0):
    return np.median(data) + np.mean(np.tanh((data - np.median(data)) / c)) * c


def shift_estimators(data):
    estimates = {
        "mean": np.mean(data),
        "median": np.median(data),
        "mle": mle_estimate(data),
    }
    for alpha in [0.05, 0.10, 0.15]:
        estimates[f"trim_mean_{alpha}"] = trim_mean(data, proportiontocut=alpha)
    for c in [0.1, 0.5, 1.0]:
        estimates[f"radical_{c}"] = radical_estimate(data, c)
    return estimates


# === Генерация выборок на основе распределения Хьюбера ===
def pure_huber(N, v=0.3, mu=0.0, sigma=1.0, random_state=None):
    huber = HuberDistribution(v)
    data = huber.sample(size=N, random_state=random_state)
    return mu + sigma * data


def contaminated_symmetric(N, v=0.3, mu=0.0, sigma=1.0, eps=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    n_cont = int(N * eps)
    n_clean = N - n_cont
    clean = pure_huber(n_clean, v, mu, sigma, random_state=rng)
    contam = pure_huber(n_cont, v, mu, sigma * 3.0, random_state=rng)
    return np.concatenate([clean, contam])


def contaminated_asymmetric(N, v=0.3, mu=0.0, sigma=1.0, eps=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    n_cont = int(N * eps)
    n_clean = N - n_cont
    clean = pure_huber(n_clean, v, mu, sigma, random_state=rng)
    contam = pure_huber(n_cont, v, mu + 3.0 * sigma, sigma * 2.0, random_state=rng)
    return np.concatenate([clean, contam])

# === Эксперимент ===
def experiment(N=500, v=0.1, eps=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    datasets = {
        "pure": pure_huber(N, v=v, mu=0.0, sigma=1.0, random_state=rng),
        "cont_sym": contaminated_symmetric(N, v=v, mu=0.0, sigma=1.0, eps=eps, random_state=rng),
        "cont_asym": contaminated_asymmetric(N, v=v, mu=0.0, sigma=1.0, eps=eps, random_state=rng),
    }

    results = {name: shift_estimators(data) for name, data in datasets.items()}
    return results, datasets


if __name__ == "__main__":
    results, datasets = experiment(N=500, v=0.1, eps=0.2, random_state=1)

    for dist_name, ests in results.items():
        print(f"\n{dist_name.upper()}:")
        for k, v in ests.items():
            print(f"  {k:15s} = {v:.4f}")

    plt.figure(figsize=(10, 6))
    for name, data in datasets.items():
        plt.hist(data, bins=50, density=True, alpha=0.5, label=name)
    plt.legend()
    plt.title("Гистограммы: Хьюбер чистое и засорённые распределения")
    plt.show()
