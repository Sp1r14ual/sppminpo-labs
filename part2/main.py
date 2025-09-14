import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
from scipy.optimize import brentq
from mpmath import mp, quad

mp.dps = 40  # точность интегралов


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


# --- Засорение распределения ---
def contaminate_data(data, contamination_fraction=0.1, shift=5.0, scale=2.0, random_state=None):
    rng = np.random.default_rng(random_state)
    contaminated = data.copy()
    n = len(data)
    m = int(contamination_fraction * n)
    idx = rng.choice(n, size=m, replace=False)
    contaminated[idx] = shift + scale * rng.normal(size=m)
    return contaminated


# --- Выборочные характеристики ---
def sample_statistics(data):
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    skewness = skew(data)
    kurt = kurtosis(data)  # эксцесс
    median = np.median(data)
    trimmed_mean = np.mean(np.sort(data)[len(data)//10:-len(data)//10])  # 10% усечение
    return dict(mean=mean, var=var, skewness=skewness, kurt=kurt,
                median=median, trimmed_mean=trimmed_mean)


# === Пример использования ===
if __name__ == "__main__":
    huber = HuberDistribution(v=0.1)
    pure = huber.sample(size=1000000, random_state=1)
    contaminated = contaminate_data(pure, contamination_fraction=0.1, shift=5, scale=2)

    print("Выборочные характеристики (чистое):", sample_statistics(pure))
    print("Выборочные характеристики (засорённое):", sample_statistics(contaminated))

    # Сравнение гистограмм
    plt.hist(pure, bins=50, density=True, alpha=0.5, label="pure")
    plt.hist(contaminated, bins=50, density=True, alpha=0.5, label="contaminated")
    xs = np.linspace(min(contaminated), max(contaminated), 500)
    plt.plot(xs, [huber.pdf(x) for x in xs], "r-", lw=2, label="теоретическая pdf")
    plt.legend()
    plt.show()
