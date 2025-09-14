import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from mpmath import mp, quad

mp.dps = 40  # точность для mpmath интеграции

class HuberDistribution:
    def __init__(self, v: float):
        if not (0.0 < v < 1.0):
            raise ValueError("v must be in (0,1)")
        self.v = float(v)

    def _solve_k(self) -> float:
        """Надёжное численное решение уравнения для k (на отрезке (0, 20])."""
        def eq(k):
            return 2.0 * norm.pdf(k) / k + 2.0 * (norm.cdf(k) - 1.0) - self.v / (1.0 - self.v)
        # brentq требует, чтобы на концах был знакопеременный (ищем в фиксированном интервале)
        a = 1e-8
        b = 20.0
        return float(brentq(eq, a, b))

    def k(self) -> float:
        return self._solve_k()

    def pdf(self, x: float) -> float:
        """Плотность Хьюбера f(x, v)."""
        k = self.k()
        coeff = (1.0 - self.v) / np.sqrt(2.0 * np.pi)
        ax = abs(x)
        if ax <= k:
            return coeff * np.exp(-0.5 * x * x)
        else:
            return coeff * np.exp(0.5 * k * k - k * ax)

    def P(self) -> float:
        """P = ∫_{-k}^{k} f(x) dx, вычисляется численно."""
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)
        def integrand(t):
            return coeff * mp.e**(-0.5 * t**2)
        I = quad(integrand, [-k, k])
        return float(I)

    def _moment(self, n: int) -> float:
        """m_n = ∫ x^n f(x) dx, считаем через 2*∫_{0}^{∞} с разбиением на [0,k] и [k,∞)."""
        k = mp.mpf(self.k())
        v_mp = mp.mpf(self.v)
        coeff = (1 - v_mp) / mp.sqrt(2 * mp.pi)

        def integrand_central(t):
            return (t**n) * coeff * mp.e**(-0.5 * t**2)
        def integrand_tail(t):
            return (t**n) * coeff * mp.e**(0.5 * k**2 - k * t)

        I1 = quad(integrand_central, [mp.mpf('0.0'), k])
        I2 = quad(integrand_tail, [k, mp.inf])
        return float(2.0 * (I1 + I2))

    def variance(self) -> float:
        """В силу симметрии мат.ожидание = 0, дисперсия = второй момент."""
        return self._moment(2)

    def excess_kurtosis(self) -> float:
        m2 = self._moment(2)
        m4 = self._moment(4)
        return m4 / (m2 * m2) - 3.0

    def sample(self, size=1, random_state=None):
        """
        Генерация выборки:
         - r1~U(0,1). Если r1<=P -> берём нормальную X и проверяем |X|<=k
         - иначе: r2~U(0,1), x2 = k - ln(r2)/k, знак по r1 vs (1+P)/2
        Для нормалей используем Box-Muller (парно) для эффективности.
        """
        rng = np.random.default_rng(random_state)
        k = float(self.k())
        P = float(self.P())
        samples = []
        normal_buffer = []

        def next_normal():
            if normal_buffer:
                return normal_buffer.pop()
            u1, u2 = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
            r = np.sqrt(-2.0 * np.log(u1))
            z1 = r * np.cos(2.0 * np.pi * u2)
            z2 = r * np.sin(2.0 * np.pi * u2)
            normal_buffer.append(z2)
            return z1

        for _ in range(size):
            r1 = rng.uniform(0.0, 1.0)
            if r1 <= P:
                # центральная нормальная часть с отсечением
                while True:
                    x1 = next_normal()
                    if abs(x1) <= k:
                        samples.append(x1)
                        break
            else:
                r2 = rng.uniform(0.0, 1.0)
                x2 = k - np.log(r2) / k
                if r1 <= (1.0 + P) / 2.0:
                    samples.append(x2)
                else:
                    samples.append(-x2)
        return np.array(samples)

# === Пример использования ===
if __name__ == "__main__":
    v = 0.01
    huber = HuberDistribution(v=v)
    print(f"v = {huber.v}")
    print(f"k = {huber.k():.4f}")
    print(f"σ² = {huber.variance():.4f}")
    print(f"γ₂ = {huber.excess_kurtosis():.4f}")
    print(f"P = {huber.P():.4f}")

    data = huber.sample(size=100000, random_state=42)
    
    # строим гистограмму (нормированную)
    plt.hist(data, bins=100, density=True, alpha=0.5, label="sample")

    # теоретическая плотность
    xs = np.linspace(min(data), max(data), 500)
    ys = [huber.pdf(x) for x in xs]
    plt.plot(xs, ys, "r-", linewidth=2, label="pdf")

    plt.title(f"Huber distribution (v={v})")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()