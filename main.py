import math
import csv
from dataclasses import dataclass
from typing import List, Tuple

# Геометрия задачи (координаты на оси x)
A_x = 0.0
B_x = 100.0
M = [200.0, 500.0, 1000.0]
N = [300.0, 600.0, 1100.0]

I = 1.0  # ток, А

# Предвычислим геометрические коэффициенты K_i,
# чтобы V_i(σ) = K_i / σ и dV_i/dσ = -K_i / σ^2
def geom_factors(Ax: float, Bx: float, M: List[float], N: List[float], I: float) -> List[float]:
    K = []
    for Mi, Ni in zip(M, N):
        term = (1.0/abs(Mi-Bx) - 1.0/abs(Mi-Ax)) - (1.0/abs(Ni-Bx) - 1.0/abs(Ni-Ax))
        K.append(I / (2.0 * math.pi) * term)
    return K

K = geom_factors(A_x, B_x, M, N, I)

def V_of_sigma(sigma: float, K: List[float]) -> List[float]:
    return [k / sigma for k in K]

def dV_dsigma(sigma: float, K: List[float]) -> List[float]:
    return [-k / (sigma**2) for k in K]

def phi(sigma: float, K: List[float], Vbar: List[float], w: List[float]) -> float:
    Vi = V_of_sigma(sigma, K)
    return sum((wi * (v - vb))**2 for wi, v, vb in zip(w, Vi, Vbar))

@dataclass
class GNResult:
    iter: int
    sigma: float
    phi: float
    delta_sigma: float

def gauss_newton_1param(
    sigma0: float,
    K: List[float],
    Vbar: List[float],
    w: List[float],
    tol: float = 1e-12,
    itmax: int = 50
) -> Tuple[float, List[GNResult]]:
    hist: List[GNResult] = []
    sigma = sigma0
    hist.append(GNResult(0, sigma, phi(sigma, K, Vbar, w), 0.0))
    for it in range(1, itmax + 1):
        dVi = dV_dsigma(sigma, K)                     # размер 3
        a11 = sum((wi * d)**2 for wi, d in zip(w, dVi))
        b1 = -sum((wi**2) * d * (v - vb)
                  for wi, d, v, vb in zip(w, dVi, V_of_sigma(sigma, K), Vbar))
        delta = b1 / a11
        sigma = sigma + delta
        hist.append(GNResult(it, sigma, phi(sigma, K, Vbar, w), delta))
        if abs(delta) < tol:
            break
    return sigma, hist

# Синтетические "измеренные" данные в примере: Vbar = K / sigma_true
sigma_true = 0.1
Vbar = V_of_sigma(sigma_true, K)

# Веса по примеру: w_i = 1 / Vbar_i
w = [1.0 / vb for vb in Vbar]

# Запуск с σ0 = 0.01
sigma0 = 0.01
sigma_est, history = gauss_newton_1param(sigma0, K, Vbar, w)

# Сохранить журнал итераций в CSV
with open("gn_iterations.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iter", "sigma", "phi", "delta_sigma"])
    for r in history:
        writer.writerow([r.iter, f"{r.sigma:.12e}", f"{r.phi:.12e}", f"{r.delta_sigma:.12e}"])

print(f"Estimated sigma = {sigma_est:.12f} S/m")
print("Wrote gn_iterations.csv with columns: iter,sigma,phi,delta_sigma")
