import numpy as np
from scipy.stats import norm
from mpmath import findroot, quad


class HuberDistribution:
    def __init__(self, v: float):
        """
        Распределение Хьюбера с параметром формы v (0 < v < 1).
        """
        if not (0.0 < v < 1.0):
            raise ValueError("Параметр v должен быть в интервале (0, 1).")
        self.v = float(v)

    def _solve_k(self) -> float:
        """
        Решение уравнения для k:
        2/k * φ(k) + 2*(Φ(k) - 1) = v/(1-v)
        """
        def equation(k):
            k = float(k)
            return 2.0 * norm.pdf(k) / k + 2.0 * (norm.cdf(k) - 1.0) - self.v / (1.0 - self.v)

        return float(findroot(equation, 1.0))

    def k(self) -> float:
        return self._solve_k()

    def pdf(self, x: float) -> float:
        """
        Плотность распределения Хьюбера f(x, v).
        """
        k = self.k()
        coeff = (1.0 - self.v) / np.sqrt(2.0 * np.pi)
        if abs(x) <= k:
            return coeff * np.exp(-0.5 * x**2)
        else:
            return coeff * np.exp(0.5 * k**2 - k * abs(x))

    def variance(self) -> float:
        k = self.k()
        phi_k = norm.pdf(k)
        Phi_k = norm.cdf(k)
        return 1.0 + (2.0 * phi_k * (k**2 + 2.0)) / (2.0 * k * phi_k + k**3 * (2.0 * Phi_k - 1.0))

    def excess_kurtosis(self) -> float:
        k = self.k()
        phi_k = norm.pdf(k)
        Phi_k = norm.cdf(k)
        sigma2 = self.variance()

        denom = (2.0 * phi_k / k + 2.0 * Phi_k - 1.0)
        num = 3.0 * (2.0 * Phi_k - 1.0) + 2.0 * phi_k * (24.0 / k**5 + 24.0 / k**3 + 12.0 / k + k)
        return num / (sigma2**2 * denom) - 3.0

    def P(self) -> float:
        """
        Вероятность попадания в центральный интервал [-k, k].
        Вычисляется как интеграл pdf(x) по этому интервалу.
        """
        k = self.k()
        return float(quad(lambda t: self.pdf(float(t)), [-k, k]))

    def sample(self, size=1, random_state=None):
        """
        Генерация выборки из распределения Хьюбера.
        """
        rng = np.random.default_rng(random_state)
        k = self.k()
        P = self.P()

        samples = []
        for _ in range(size):
            r1 = rng.uniform(0.0, 1.0)
            if r1 <= P:
                # Центральная часть
                while True:
                    x1 = rng.normal()
                    if abs(x1) <= k:
                        samples.append(x1)
                        break
            else:
                # Хвосты
                r2 = rng.uniform(0.0, 1.0)
                x2 = k - np.log(r2) / k
                if r1 <= (1.0 + P) / 2.0:
                    samples.append(x2)
                else:
                    samples.append(-x2)

        return np.array(samples)


# === Пример использования ===
if __name__ == "__main__":
    huber = HuberDistribution(v=0.01)
    print(f"v = {huber.v}")
    print(f"k = {huber.k():.4f}")
    print(f"σ² = {huber.variance():.4f}")
    print(f"γ₂ = {huber.excess_kurtosis():.4f}")
    print(f"P = {huber.P():.4f}")

    data = huber.sample(size=10, random_state=42)
    print("Пример выборки:", data)
