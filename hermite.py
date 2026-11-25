# Interpolación de Hermite con diferencias divididas (nodos repetidos)
#     H(x_i) = f(x_i),  H'(x_i) = f'(x_i)
# Pasos (como en la guía):
#   1) z = (x_0, x_0, x_1, x_1, ..., x_n, x_n)
#   2) Q[2i][0] = Q[2i+1][0] = f(x_i)
#   3) Orden 1: si z_k = z_{k-1} se usa f'(x_i); si no, la diferencia dividida usual
#   4) Orden j >= 2: Q[k][j] = (Q[k][j-1] - Q[k-1][j-1]) / (z_k - z_{k-j})
#   5) Coeficientes a_k = Q[k][k] (diagonal)
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x = sp.symbols("x")
xi  = [0, 1, 2]            # nodos (distintos)
fi  = [0, 1, 32]           # valores f(x_i)
dfi = [0, 5, 80]           # derivadas f'(x_i)
# Si tienes una función, por ejemplo f(x) = sin(x):
#   f = sp.sin(x)
#   fi  = [f.subs(x, v) for v in xi]
#   dfi = [sp.diff(f, x).subs(x, v) for v in xi]

x_eval = None              # punto donde quieres evaluar H(x) (o None)

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
xi  = [a_numero(v) for v in xi]
fi  = [a_numero(v) for v in fi]
dfi = [a_numero(v) for v in dfi]
n = len(xi)
if not (len(fi) == len(dfi) == n):
    raise ValueError("xi, fi y dfi deben tener la misma cantidad de datos.")
if len(set(xi)) != n:
    raise ValueError("Los nodos xi deben ser distintos.")

m = 2*n                     # filas de la tabla
z = [None]*m
Q = [[None]*m for _ in range(m)]

# Pasos 1 y 2: duplicar nodos y valores
for i in range(n):
    z[2*i] = z[2*i + 1] = xi[i]
    Q[2*i][0] = Q[2*i + 1][0] = fi[i]

# Paso 3: orden 1
for i in range(n):
    Q[2*i + 1][1] = dfi[i]                                  # nodo repetido -> derivada
    if i > 0:
        Q[2*i][1] = (Q[2*i][0] - Q[2*i - 1][0])/(z[2*i] - z[2*i - 1])

# Paso 4: órdenes 2, 3, ...
for j in range(2, m):
    for k in range(j, m):
        Q[k][j] = (Q[k][j-1] - Q[k-1][j-1])/(z[k] - z[k-j])

# Paso 5: coeficientes y polinomio
coef = [Q[k][k] for k in range(m)]
H = 0
producto = 1
terminos = []
for k in range(m):
    H = H + coef[k]*producto
    if coef[k] != 0:
        terminos.append(f"({texto(coef[k])})" + (f"*{texto(producto)}" if k > 0 else ""))
    producto = producto*(x - z[k])
H_expandido = sp.expand(H)
dH = sp.diff(H_expandido, x)

# SALIDA
ancho = max(9, max(len(texto(v)) for fila in Q for v in fila) + 2)
print("Tabla de Hermite (diferencias divididas con nodos repetidos)")
print(f"{'i':>3} {'z_i':>{ancho}}" + "".join(f"{'Orden ' + str(j):>{ancho}}" for j in range(m)))
for k in range(m):
    print(f"{k:>3} {texto(z[k]):>{ancho}}" + "".join(f"{texto(Q[k][j]):>{ancho}}" for j in range(m)))

print("\nCoeficientes (diagonal):", ", ".join(texto(c) for c in coef))
print("\nForma de Newton:")
print("H(x) = " + (" + ".join(terminos) if terminos else "0"))
print("\nForma desarrollada:")
print("H(x) =", texto(H_expandido))
print("Grado:", sp.degree(H_expandido, x) if H_expandido != 0 else 0, f"(máximo posible: {m - 1})")

print("\nComprobación:")
for i in range(n):
    print(f"  H({texto(xi[i])}) = {texto(H_expandido.subs(x, xi[i]))}  (f = {texto(fi[i])}),   "
          f"H'({texto(xi[i])}) = {texto(dH.subs(x, xi[i]))}  (f' = {texto(dfi[i])})")

if x_eval is not None:
    valor = H_expandido.subs(x, sp.sympify(x_eval))
    print(f"\nH({x_eval}) = {texto(valor)}  ≈ {float(valor)}")

# GRÁFICA
hx = sp.lambdify(x, H_expandido, "numpy")
xs = np.linspace(float(min(xi)), float(max(xi)), 400)
plt.plot(xs, hx(xs) * np.ones_like(xs), label="H(x) de Hermite")
plt.plot([float(v) for v in xi], [float(v) for v in fi], "ko", label="datos")
# Pequeñas rectas tangentes con la pendiente f'(x_i)
ancho_tan = (float(max(xi)) - float(min(xi)))/10
for i in range(n):
    x0, y0, d0 = float(xi[i]), float(fi[i]), float(dfi[i])
    plt.plot([x0 - ancho_tan, x0 + ancho_tan], [y0 - d0*ancho_tan, y0 + d0*ancho_tan], "r-",
             label="pendientes f'(x_i)" if i == 0 else None)
if x_eval is not None:
    plt.plot(float(x_eval), float(valor), "m*", markersize=12, label=f"H({x_eval})")
plt.title("Interpolación de Hermite")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.legend()
plt.grid(True)
plt.show()
