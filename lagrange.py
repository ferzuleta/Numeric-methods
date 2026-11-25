# Interpolación: Polinomio de Lagrange
#     P(x) = sum_k f(x_k) * L_k(x),   L_k(x) = prod_{j != k} (x - x_j) / (x_k - x_j)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x = sp.symbols("x")
xi = [0, 1, 2, 4]          # nodos (distintos)
fi = [1, 3, 2, 5]          # valores f(x_i)
# Si tienes una función en vez de una tabla, por ejemplo:
#   fi = [sp.exp(v) for v in xi]

x_eval = 3                 # punto donde quieres evaluar P(x) (o None)

# PROCEDIMIENTO
def a_numero(v):
    """Enteros y textos como "1/3" -> fracción exacta; decimales, exp(2), sqrt(2)... -> decimal."""
    v = sp.sympify(v)
    return v if v.is_Rational else sp.N(v)

def texto(v):
    """Muestra fracciones exactas y decimales con 6 cifras significativas."""
    if v is None:
        return ""
    v = sp.sympify(v)
    return sp.sstr(v.xreplace({f: sp.Float(f, 6) for f in v.atoms(sp.Float)}), full_prec=False)

# Convertir los datos
xi = [a_numero(v) for v in xi]
fi = [a_numero(v) for v in fi]
n = len(xi)
if len(fi) != n:
    raise ValueError("xi y fi deben tener la misma cantidad de datos.")
if len(set(xi)) != n:
    raise ValueError("Los nodos xi deben ser distintos.")

# Polinomios base L_k(x)
L = []
for k in range(n):
    Lk = 1
    for j in range(n):
        if j != k:
            Lk = Lk*(x - xi[j])/(xi[k] - xi[j])
    L.append(Lk)

P = sum(fi[k]*L[k] for k in range(n))
P_expandido = sp.expand(P)

# SALIDA
print("Polinomios base de Lagrange:")
for k in range(n):
    print(f"  L_{k}(x) = {texto(sp.factor(L[k]))}")
print()
print("P(x) = " + " + ".join(f"({texto(fi[k])})*L_{k}(x)" for k in range(n)))
print("P(x) =", texto(P_expandido))
print("Grado:", sp.degree(P_expandido, x) if P_expandido != 0 else 0)

print("\nComprobación P(x_i) = f(x_i):")
for k in range(n):
    print(f"  P({texto(xi[k])}) = {texto(P_expandido.subs(x, xi[k]))}   (f = {texto(fi[k])})")

if x_eval is not None:
    valor = P_expandido.subs(x, sp.sympify(x_eval))
    print(f"\nP({x_eval}) = {texto(valor)}  ≈ {float(valor)}")

# GRÁFICA
px = sp.lambdify(x, P_expandido, "numpy")
xs = np.linspace(float(min(xi)), float(max(xi)), 400)
plt.plot(xs, px(xs) * np.ones_like(xs), label="P(x) de Lagrange")
plt.plot([float(v) for v in xi], [float(v) for v in fi], "ko", label="datos")
if x_eval is not None:
    plt.plot(float(x_eval), float(valor), "r*", markersize=12, label=f"P({x_eval})")
plt.title("Polinomio de Lagrange")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.legend()
plt.grid(True)
plt.show()
