# Sistemas de ecuaciones no lineales: Método de Punto Fijo
# Para resolver F(X) = 0 se reescribe como X = G(X) y se itera
#     X^(k+1) = G(X^(k))      (todas las variables se actualizan SIMULTÁNEAMENTE)
# hasta que ||X^(k+1) - X^(k)||_inf < tol.
#
# Convergencia: si cerca de la solución p la matriz jacobiana de G cumple
#     ||J_G(p)||_inf < 1   (o su radio espectral rho(J_G(p)) < 1)
# la iteración converge, y el error se reduce aproximadamente por un factor rho
# en cada iteración:  e_(k+1) ≈ rho * e_k.  Mientras más pequeño rho, más rápido.
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# INGRESO
x, y = sp.symbols("x y")
variables = [x, y]

# Sistema original F(X) = 0 (se usa para comprobar la solución)
ecuaciones = [
    x**2 + 3*x - y + sp.log(1 + y) - 9,
    y**2 + 2*y - x + sp.exp(-x) - 4,
]

# Reescrituras X = G(X): una lista [g1, g2, ...] por cada reescritura
reescrituras = {
    # x(x + 3) = 9 + y - ln(1+y)    y    y(y + 2) = 4 + x - e^(-x)
    "R1 (factorizando)": [
        (9 + y - sp.log(1 + y))/(x + 3),
        (4 + x - sp.exp(-x))/(y + 2),
    ],
    # Completando cuadrados: (x + 3/2)^2 = 45/4 + y - ln(1+y)   y   (y + 1)^2 = 5 + x - e^(-x)
    "R2 (completando cuadrados)": [
        (-3 + sp.sqrt(45 + 4*y - 4*sp.log(1 + y)))/2,
        -1 + sp.sqrt(5 + x - sp.exp(-x)),
    ],
    # Otras reescrituras para probar (quita el # para activarlas):
    # "R3 (raíz cuadrada)": [sp.sqrt(9 - 3*x + y - sp.log(1 + y)), sp.sqrt(4 - 2*y + x - sp.exp(-x))],
    # "R4 (despeje lineal, DIVERGE)": [(9 + y - sp.log(1 + y) - x**2)/3, (4 + x - sp.exp(-x) - y**2)/2],
}

X0 = [1, 1]          # aproximación inicial
tol = 1e-4           # criterio de detención ||X^(k+1) - X^(k)||_inf < tol
max_iter = 200

# PROCEDIMIENTO
nombres = [str(v) for v in variables]
F = sp.lambdify(variables, ecuaciones, "numpy")


def norma_inf_jacobiano(J, punto):
    """Devuelve ||J||_inf (máxima suma de filas) y el radio espectral de J en el punto."""
    Jp = np.array(J.subs(dict(zip(variables, punto))).evalf(), dtype=float)
    return float(np.max(np.sum(np.abs(Jp), axis=1))), float(np.max(np.abs(np.linalg.eigvals(Jp))))


def punto_fijo_sistema(G):
    """Itera X^(k+1) = G(X^(k)) actualizando todas las variables a la vez."""
    g = sp.lambdify(variables, G, "numpy")
    X = np.array(X0, dtype=float)
    historial = [X.copy()]
    errores = [np.nan]                                   # la iteración 0 no tiene error
    estado = f"no alcanzó la tolerancia en {max_iter} iteraciones"
    for k in range(max_iter):
        with np.errstate(all="ignore"):
            Xnuevo = np.array(g(*X), dtype=float)
        if not np.all(np.isfinite(Xnuevo)):
            estado = f"FALLA en la iteración {k + 1}: G(X) no está definida (raíz de negativo, log de negativo, ...)"
            break
        error = float(np.max(np.abs(Xnuevo - X)))       # norma infinito
        historial.append(Xnuevo)
        errores.append(error)
        X = Xnuevo
        if error < tol:
            estado = "converge"
            break
        if error > 1e10:
            estado = f"DIVERGE (los valores crecen sin control en la iteración {k + 1})"
            break
    return np.array(historial), np.array(errores), estado


resultados = {}
for nombre, G in reescrituras.items():
    print("=" * 78)
    print(nombre)
    print("=" * 78)
    for v, gi in zip(variables, G):
        print(f"  {v} = {gi}")

    historial, errores, estado = punto_fijo_sistema(G)

    # Tabla de iteraciones
    ancho = 14
    encabezado = f"{'k':>4}" + "".join(f"{v + '^(k)':>{ancho}}" for v in nombres) \
                 + "".join(f"{'|Δ' + v + '|':>{ancho}}" for v in nombres) + f"{'||ΔX||_inf':>{ancho}}"
    print("\n" + encabezado)
    print("-" * len(encabezado))
    for k in range(len(historial)):
        fila = f"{k:>4}" + "".join(f"{val:>{ancho}.8f}" for val in historial[k])
        if k == 0:
            fila += "".join(f"{'—':>{ancho}}" for _ in nombres) + f"{'—':>{ancho}}"
        else:
            dif = np.abs(historial[k] - historial[k - 1])
            fila += "".join(f"{d:>{ancho}.3e}" for d in dif) + f"{errores[k]:>{ancho}.3e}"
        print(fila)

    # Resultado y análisis de convergencia
    J = sp.Matrix(G).jacobian(variables)
    Xf = historial[-1]
    print(f"\nEstado: {estado}  ({len(historial) - 1} iteraciones)")
    if estado == "converge":
        print("Solución aproximada: " + ", ".join(f"{v} = {val:.8f}" for v, val in zip(nombres, Xf)))
        residuo = np.array(F(*Xf), dtype=float)
        print("Comprobación F(X) = [" + ", ".join(f"{r:.2e}" for r in residuo) + "]")
        nJ, rho = norma_inf_jacobiano(J, Xf)
        print(f"||J_G||_inf en la solución = {nJ:.4f},   radio espectral rho = {rho:.4f}")
        e = errores[1:]
        razones = e[1:]/e[:-1]
        if razones.size:
            print(f"Razón de errores e_(k+1)/e_k (últimas): " + ", ".join(f"{r:.4f}" for r in razones[-4:]))
    else:
        nJ, rho = norma_inf_jacobiano(J, [float(v) for v in X0])
        print(f"||J_G||_inf en X0 = {nJ:.4f}  (>= 1 indica que no hay garantía de convergencia)")
    print()
    resultados[nombre] = dict(historial=historial, errores=errores, estado=estado,
                              iteraciones=len(historial) - 1, normaJ=nJ, rho=rho)

# COMPARACIÓN (parte b)
print("=" * 78)
print("COMPARACIÓN DE LOS ERRORES ||X^(k+1) - X^(k)||_inf")
print("=" * 78)
nombres_r = list(resultados)
ancho = max(16, max(len(n) for n in nombres_r) + 2)
print(f"{'k':>4}" + "".join(f"{n:>{ancho}}" for n in nombres_r))
kmax = max(r["iteraciones"] for r in resultados.values())
for k in range(1, kmax + 1):
    fila = f"{k:>4}"
    for n in nombres_r:
        e = resultados[n]["errores"]
        fila += f"{e[k]:>{ancho}.3e}" if k < len(e) else f"{'':>{ancho}}"
    print(fila)

print("\nResumen:")
for n, r in resultados.items():
    print(f"  {n}: {r['estado']}, {r['iteraciones']} iteraciones, "
          f"||J_G||_inf = {r['normaJ']:.4f}, rho = {r['rho']:.4f}")

convergentes = {n: r for n, r in resultados.items() if r["estado"] == "converge"}
if convergentes:
    mejor = min(convergentes, key=lambda n: (convergentes[n]["iteraciones"], convergentes[n]["rho"]))
    r = convergentes[mejor]
    print(f"\nConclusión: {mejor} presenta mejor comportamiento de convergencia.")
    print(f"  Necesita {r['iteraciones']} iteraciones y su jacobiano en la solución tiene "
          f"||J_G||_inf = {r['normaJ']:.4f} (rho = {r['rho']:.4f}).")
    print("  En punto fijo la convergencia es lineal: el error se multiplica aproximadamente por rho")
    print("  en cada iteración. Mientras más pequeña es la norma del jacobiano de G, más rápido")
    print("  disminuye el error y menos iteraciones se necesitan.")
    for n, otro in convergentes.items():
        if n != mejor:
            print(f"  {n} tiene rho = {otro['rho']:.4f}, por eso su error baja más lento "
                  f"({otro['iteraciones']} iteraciones).")

# GRÁFICA: error vs iteración (escala logarítmica)
for n, r in resultados.items():
    e = r["errores"][1:]
    plt.semilogy(range(1, len(e) + 1), e, "o-", label=f"{n} ({r['iteraciones']} it.)")
plt.axhline(tol, color="k", linestyle="--", label=f"tol = {tol:g}")
plt.title("Punto fijo para sistemas: error por iteración")
plt.xlabel("iteración k")
plt.gca().xaxis.get_major_locator().set_params(integer=True)
plt.ylabel(r"$\|X^{(k)} - X^{(k-1)}\|_\infty$")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.show()
