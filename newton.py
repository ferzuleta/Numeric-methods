# Interpolación: Polinomio de Newton con diferencias divididas
#     P(x) = a_0 + a_1 (x - x_0) + a_2 (x - x_0)(x - x_1) + ...
#     a_k = f[x_0, ..., x_k]  (diagonal de la tabla de diferencias divididas)
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

# Tabla de diferencias divididas: Q[i][j] = f[x_{i-j}, ..., x_i]
Q = [[None]*n for _ in range(n)]
for i in range(n):
    Q[i][0] = fi[i]
for j in range(1, n):
    for i in range(j, n):
        Q[i][j] = (Q[i][j-1] - Q[i-1][j-1])/(xi[i] - xi[i-j])

# Coeficientes = diagonal
coef = [Q[k][k] for k in range(n)]

# Polinomio en forma de Newton
P = 0
producto = 1
terminos = []
for k in range(n):
    P = P + coef[k]*producto
    terminos.append(f"({texto(coef[k])})" + (f"*{texto(producto)}" if k > 0 else ""))
    producto = producto*(x - xi[k])
P_expandido = sp.expand(P)

# SALIDA
ancho = max(10, max(len(texto(v)) for fila in Q for v in fila) + 2)
print("Tabla de diferencias divididas")
print(f"{'i':>3} {'x_i':>{ancho}}" + "".join(f"{'Orden ' + str(j):>{ancho}}" for j in range(n)))
for i in range(n):
    print(f"{i:>3} {texto(xi[i]):>{ancho}}" + "".join(f"{texto(Q[i][j]):>{ancho}}" for j in range(n)))

print("\nCoeficientes (diagonal):", ", ".join(texto(c) for c in coef))
print("\nForma de Newton:")
print("P(x) = " + " + ".join(terminos))
print("\nForma desarrollada:")
print("P(x) =", texto(P_expandido))

print("\nComprobación P(x_i) = f(x_i):")
for k in range(n):
    print(f"  P({texto(xi[k])}) = {texto(P_expandido.subs(x, xi[k]))}   (f = {texto(fi[k])})")

if x_eval is not None:
    valor = P_expandido.subs(x, sp.sympify(x_eval))
    print(f"\nP({x_eval}) = {texto(valor)}  ≈ {float(valor)}")

# GRÁFICA
px = sp.lambdify(x, P_expandido, "numpy")
xs = np.linspace(float(min(xi)), float(max(xi)), 400)
plt.plot(xs, px(xs) * np.ones_like(xs), label="P(x) de Newton")
plt.plot([float(v) for v in xi], [float(v) for v in fi], "ko", label="datos")
if x_eval is not None:
    plt.plot(float(x_eval), float(valor), "r*", markersize=12, label=f"P({x_eval})")
plt.title("Polinomio de Newton (diferencias divididas)")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.legend()
plt.grid(True)
plt.show()
