# Ecuaciones no lineales: Método de Punto Fijo
# Para resolver f(x) = 0 se reescribe como x = g(x) y se itera
#     x_{n+1} = g(x_n)
# hasta que |x_{n+1} - x_n| < tol.
#
# Teorema (condiciones suficientes de convergencia en [a, b]):
#   1) g(x) está en [a, b] para todo x en [a, b]   -> existe un punto fijo
#   2) |g'(x)| <= k < 1 para todo x en [a, b]       -> es único y la iteración converge
#   Cota del error:  |x_n - p| <= k^n * max(x_0 - a, b - x_0)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x = sp.symbols("x")
fx = x**3 + 4*x**2 - 10          # ecuación f(x) = 0 (solo para comprobar; puede ser None)
gx = sp.sqrt(10/(x + 4))         # función de iteración: x = g(x)

a = 1                            # intervalo [a, b] donde se busca la raíz
b = 2
x0 = 1.5                         # valor inicial
tol = 1e-6                       # tolerancia
max_iter = 100                   # máximo de iteraciones

# PROCEDIMIENTO
g = sp.lambdify(x, gx, "numpy")
dgx = sp.diff(gx, x)
dg = sp.lambdify(x, dgx, "numpy")

# --- 1) Revisar las condiciones de convergencia en [a, b] ---
malla = np.linspace(a, b, 10001)
with np.errstate(all="ignore"):
    g_malla = g(malla) * np.ones_like(malla)
    dg_malla = np.abs(dg(malla) * np.ones_like(malla))

print("g(x)  =", gx)
print("g'(x) =", sp.simplify(dgx))
print(f"\nCondiciones de convergencia en [{a}, {b}]:")

cond1 = bool(np.all(np.isfinite(g_malla)) and np.all((g_malla >= a) & (g_malla <= b)))
print(f"  1) g(x) en [a, b]:  min g = {np.nanmin(g_malla):.6g}, max g = {np.nanmax(g_malla):.6g}",
      " -> SE CUMPLE" if cond1 else " -> NO se cumple")

k = float(np.nanmax(dg_malla))
cond2 = k < 1
print(f"  2) k = max|g'(x)| = {k:.8g}", " -> SE CUMPLE (k < 1)" if cond2 else " -> NO se cumple (k >= 1)")

if cond1 and cond2:
    if k > 0:
        n_teorico = int(np.ceil(np.log(tol/max(x0 - a, b - x0))/np.log(k)))
        print(f"  Iteraciones suficientes según la cota k^n·max(x0-a, b-x0) < tol: n >= {max(n_teorico, 1)}")
    print("  => La iteración converge a un único punto fijo en [a, b].")
else:
    print("  => No se garantiza la convergencia (puede converger igual, o diverger).")

# --- 2) Iterar x_{n+1} = g(x_n) ---
print(f"\n{'n':>4} {'x_n':>16} {'g(x_n)':>16} {'|x_n+1 - x_n|':>16}")
xn = float(x0)
iteraciones = [xn]
convergio = False
for n in range(max_iter):
    with np.errstate(all="ignore"):
        xnuevo = float(g(xn))
    if not np.isfinite(xnuevo):
        print(f"{n:>4} {xn:>16.10f} {'—':>16}   g(x_n) no está definido: el método falla.")
        break
    error = abs(xnuevo - xn)
    print(f"{n:>4} {xn:>16.10f} {xnuevo:>16.10f} {error:>16.3e}")
    iteraciones.append(xnuevo)
    xn = xnuevo
    if error < tol:
        convergio = True
        break
    if abs(xn) > 1e12:
        print("Los valores crecen sin control: el método DIVERGE.")
        break

# SALIDA
print()
if convergio:
    print(f"Punto fijo (raíz) aproximado: p = {xn:.10f}")
    print(f"Iteraciones realizadas: {len(iteraciones) - 1}")
    print(f"Comprobación: g(p) = {float(g(xn)):.10f}")
    if fx is not None:
        f = sp.lambdify(x, fx, "numpy")
        print(f"Comprobación: f(p) = {float(f(xn)):.3e}")
else:
    print(f"No se alcanzó la tolerancia {tol} en {len(iteraciones) - 1} iteraciones.")

# GRÁFICA: y = x, y = g(x) y el diagrama de telaraña de las iteraciones
pts = np.array(iteraciones[:40])
pts = pts[np.isfinite(pts)]
izq = min(a, pts.min()) if pts.size else a
der = max(b, pts.max()) if pts.size else b
margen = 0.1*(der - izq)
xs = np.linspace(izq - margen, der + margen, 400)
with np.errstate(all="ignore"):
    plt.plot(xs, g(xs) * np.ones_like(xs), label="y = g(x)")
plt.plot(xs, xs, "k--", label="y = x")
for i in range(len(pts) - 1):
    plt.plot([pts[i], pts[i]], [pts[i], pts[i + 1]], "r-", linewidth=0.8)      # vertical hasta g
    plt.plot([pts[i], pts[i + 1]], [pts[i + 1], pts[i + 1]], "r-", linewidth=0.8)  # horizontal hasta y = x
plt.plot(pts[0], pts[0], "go", label="x0")
if convergio:
    plt.plot(xn, xn, "r*", markersize=12, label=f"p ≈ {xn:.6f}")
plt.axvspan(a, b, color="gray", alpha=0.1, label="[a, b]")
plt.title("Método de Punto Fijo")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
