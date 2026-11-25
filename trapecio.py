# Integración: Regla de los trapecios
# El número de tramos se elige para que el error sea menor que tol:
#     |E| <= (b-a)^3 * M2 / (12 n^2) <= tol   =>   n >= sqrt((b-a)^3 * M2 / (12 tol))
# donde M2 = max |f''(x)| en [a, b].
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x = sp.symbols("x")
fx_simbolica = 1/(x**5 + 1)      # escribe la función con sympy: sp.sin(x), sp.exp(x), sp.sqrt(x), ...

# intervalo de integración
a = 1
b = 3
tol = 0.01

# PROCEDIMIENTO
# Versión numérica de f para evaluar rápido (acepta números y arreglos de numpy)
fx = sp.lambdify(x, fx_simbolica, "numpy")

# --- M2: máximo de |f''(x)| en [a, b] ---
# Se evalúa |f''| en una malla muy fina del intervalo (incluye los extremos).
# Es mucho más rápido y seguro que sp.solve(f''' = 0), que puede tardar
# minutos o no encontrar las raíces.
segunda_derivada = sp.diff(fx_simbolica, x, 2)
f2 = sp.lambdify(x, segunda_derivada, "numpy")
malla = np.linspace(a, b, 100001)
valores_f2 = np.abs(f2(malla) * np.ones_like(malla))   # *ones por si f'' es constante
k = np.argmax(valores_f2)
M2 = float(valores_f2[k])
print("f''(x) =", sp.simplify(segunda_derivada))
print("M2 = max|f''(x)| en [a, b] =", M2, " (alcanzado cerca de x =", round(float(malla[k]), 4), ")")

# Número de tramos (al menos 1, por si f'' = 0)
tramos = max(1, int(np.ceil(np.sqrt((b - a)**3 * M2 / (12 * tol)))))

# Regla del Trapecio con tramos equidistantes
h = (b - a)/tramos
xi = a + np.arange(tramos + 1)*h        # nodos x_0, ..., x_n (sin acumular errores de suma)
fi = fx(xi) * np.ones_like(xi)          # *ones por si f es constante
area = h/2*(fi[0] + 2*np.sum(fi[1:-1]) + fi[-1])

# Cota teórica del error con esos tramos
cota_error = (b - a)**3 * M2 / (12 * tramos**2)

# Valor de referencia: integral calculada numéricamente con alta precisión.
# (sp.integrate puede dar una expresión enorme o no encontrar primitiva.)
integral_ref = float(sp.Integral(fx_simbolica, (x, a, b)).evalf(20))
error_real = abs(integral_ref - area)

# SALIDA
print("tramos para Trapecio:", tramos)
print("h =", h)
print("Integral aproximada (Trapecio):", area)
print("Integral de referencia:        ", integral_ref)
print("Error real:   ", error_real)
print("Cota teórica: ", cota_error)
print("¿Error real <= tol?", error_real <= tol)

# GRÁFICA
xg = np.linspace(a, b, 400)
plt.plot(xg, fx(xg) * np.ones_like(xg), label="f(x)")
plt.fill_between(xi, 0, fi, color="tab:green", alpha=0.3, label=f"trapecios (n = {tramos})")
plt.plot(xi, fi, "o-", color="tab:green")
plt.vlines(xi, 0, fi, color="tab:green", linewidth=0.8)
plt.title("Regla de los trapecios")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
