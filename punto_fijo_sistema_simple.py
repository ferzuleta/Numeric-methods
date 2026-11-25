# Sistemas no lineales: Punto Fijo (versión simplificada)
#   x^2 + 3x - y + ln(1+y) = 9
#   y^2 + 2y - x + e^(-x)  = 4
# Reescritura: se despeja solo el término al cuadrado y se saca raíz
#   x = sqrt(9 - 3x + y - ln(1+y))
#   y = sqrt(4 - 2y + x - e^(-x))
import numpy as np

# INGRESO
g1 = lambda x, y: np.sqrt(9 - 3*x + y - np.log(1 + y))
g2 = lambda x, y: np.sqrt(4 - 2*y + x - np.exp(-x))

x0, y0 = 1, 1      # aproximación inicial X(0)
tol = 1e-4         # ||X(k+1) - X(k)||_inf < tol
max_iter = 500

# PROCEDIMIENTO
print(f"{'k':>4} {'x':>12} {'y':>12} {'|Δx|':>11} {'|Δy|':>11} {'error':>11}")
print(f"{0:>4} {x0:>12.8f} {y0:>12.8f}")
x, y = x0, y0
k = 0
error = 1
while error >= tol and k < max_iter:
    xn = g1(x, y)          # ambas variables se actualizan
    yn = g2(x, y)          # SIMULTÁNEAMENTE con los valores anteriores
    error = max(abs(xn - x), abs(yn - y))     # norma infinito
    k = k + 1
    print(f"{k:>4} {xn:>12.8f} {yn:>12.8f} {abs(xn - x):>11.3e} {abs(yn - y):>11.3e} {error:>11.3e}")
    x, y = xn, yn

# SALIDA
print()
if error < tol:
    print(f"Converge en {k} iteraciones")
    print(f"x = {x:.8f},  y = {y:.8f}")
    print("Comprobación:")
    print("  ec. 1:", x**2 + 3*x - y + np.log(1 + y), "(debe ser ≈ 9)")
    print("  ec. 2:", y**2 + 2*y - x + np.exp(-x), "(debe ser ≈ 4)")
else:
    print(f"No converge en {max_iter} iteraciones")
