# Integración: Regla de Simpson 1/3 + integral exacta con sympy
# El número de tramos (par) se elige para que el error sea menor que tol:
#     |E| <= (b-a)^5 * M4 / (180 n^4) <= tol   =>   n >= ((b-a)^5 * M4 / (180 tol))^(1/4)
# donde M4 = max |f''''(x)| en [a, b].
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x = sp.symbols("x")
fx_simbolica = (x**6 - 2*x**5)/10   # escribe la función con sympy: sp.sin(x), sp.exp(x), sp.sqrt(x), ...

# intervalo de integración
a = 0
b = 4
tol = 0.001

# Calcular también la integral exacta (forma cerrada) con sp.integrate.
# Ponlo en False si la función no tiene primitiva o si sympy tarda demasiado.
mostrar_exacta = True

# PROCEDIMIENTO
# Versión numérica de f para evaluar rápido (acepta números y arreglos de numpy)
fx = sp.lambdify(x, fx_simbolica, "numpy")

# --- M4: máximo de |f''''(x)| en [a, b] ---
# Se evalúa |f''''| en una malla muy fina del intervalo (incluye los extremos).
# Es mucho más rápido y seguro que sp.solve(f^(5) = 0), que puede tardar
# minutos o no encontrar las raíces.
cuarta_derivada = sp.diff(fx_simbolica, x, 4)
f4 = sp.lambdify(x, cuarta_derivada, "numpy")
malla = np.linspace(a, b, 100001)
valores_f4 = np.abs(f4(malla) * np.ones_like(malla))   # *ones por si f'''' es constante
k = np.argmax(valores_f4)
M4 = float(valores_f4[k])
print("f''''(x) =", sp.simplify(cuarta_derivada))
print("M4 = max|f''''(x)| en [a, b] =", M4, " (alcanzado cerca de x =", round(float(malla[k]), 4), ")")

# Número de tramos: al menos 2 y siempre PAR (Simpson 1/3 usa los tramos de a dos)
tramos = max(2, int(np.ceil(((b - a)**5 * M4 / (180 * tol))**(1/4))))
if tramos % 2 != 0:
    tramos += 1

# Regla de Simpson 1/3 con tramos equidistantes
#   integral ≈ h/3 * [ f0 + 4(f1 + f3 + ...) + 2(f2 + f4 + ...) + fn ]
h = (b - a)/tramos
xi = a + np.arange(tramos + 1)*h        # nodos x_0, ..., x_n (sin acumular errores de suma)
fi = fx(xi) * np.ones_like(xi)          # *ones por si f es constante
area = h/3*(fi[0] + 4*np.sum(fi[1:-1:2]) + 2*np.sum(fi[2:-1:2]) + fi[-1])

# Cota teórica del error con esos tramos
cota_error = (b - a)**5 * M4 / (180 * tramos**4)

# Valor de referencia: integral calculada numéricamente con alta precisión
integral_ref = float(sp.Integral(fx_simbolica, (x, a, b)).evalf(20))
error_real = abs(integral_ref - area)

# SALIDA
print("tramos para Simpson:", tramos)
print("h =", h)
print("Integral aproximada (Simpson):", area)
if mostrar_exacta:
    integral_exacta = sp.integrate(fx_simbolica, (x, a, b))
    print("Integral exacta (SymPy):      ", integral_exacta)
print("Integral exacta (decimal):    ", integral_ref)
print("Error real:   ", error_real)
print("Cota teórica: ", cota_error)
print("¿Error real <= tol?", error_real <= tol)

# GRÁFICA: cada par de tramos se aproxima con una parábola
xg = np.linspace(a, b, 400)
plt.plot(xg, fx(xg) * np.ones_like(xg), "k", linewidth=2, label="f(x)")
for i in range(0, tramos, 2):
    xp = xi[i:i+3]
    yp = fi[i:i+3]
    coef = np.polyfit(xp, yp, 2)                  # parábola por los 3 puntos
    xs = np.linspace(xp[0], xp[-1], 30)
    plt.fill_between(xs, 0, np.polyval(coef, xs), color="tab:orange", alpha=0.3,
                     label=f"parábolas (n = {tramos})" if i == 0 else None)
plt.plot(xi, fi, "o", color="tab:orange")
plt.vlines(xi, 0, fi, color="tab:orange", linewidth=0.8)
plt.title("Regla de Simpson 1/3")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
