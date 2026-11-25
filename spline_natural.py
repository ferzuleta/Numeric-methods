"""
Spline cúbico natural
=====================

Uso rápido:

    from spline_natural import spline_natural

    s = spline_natural([0, 1, 2], [0, 1, 0])
    s.mostrar()          # imprime los tramos S_i(x) y la tabla de coeficientes
    s.verificar()        # revisa interpolación, empalmes de S, S', S'' y condiciones naturales
    print(s(0.5))        # evalúa el spline en x = 0.5
    s.graficar()         # dibuja el spline (requiere matplotlib)

En cada intervalo [x_i, x_{i+1}] el spline es

    S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3

con las condiciones naturales S''(x_0) = 0 y S''(x_n) = 0.

Si los datos son enteros o fracciones, los coeficientes se calculan en forma
EXACTA (con fracciones, p. ej. 3/2 en vez de 1.5). Si hay decimales, se usan
números de punto flotante.

También se puede ejecutar directamente:

    python3 spline_natural.py
"""

from fractions import Fraction


# ----------------------------------------------------------------------------
# Construcción del spline (algoritmo clásico, sistema tridiagonal)
# ----------------------------------------------------------------------------

def _a_numero(v):
    """Convierte enteros/fracciones a Fraction (exacto); deja los float como float."""
    if isinstance(v, (int, Fraction)):
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(v)          # permite escribir "1/3"
    return float(v)


def spline_natural(x, y):
    """
    Construye el spline cúbico natural que pasa por los puntos (x_i, y_i).

    x : lista de nodos (deben estar en orden creciente)
    y : lista de valores f(x_i)

    Devuelve un objeto SplineNatural.
    """
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma cantidad de datos.")
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 puntos.")

    x = [_a_numero(v) for v in x]
    y = [_a_numero(v) for v in y]
    n = len(x) - 1                      # número de tramos

    for i in range(n):
        if x[i + 1] <= x[i]:
            raise ValueError("Los nodos x deben ser distintos y estar en orden creciente.")

    # Paso 1: longitudes de los intervalos h_i = x_{i+1} - x_i
    h = [x[i + 1] - x[i] for i in range(n)]

    # Paso 2: lado derecho del sistema
    alfa = [0] * (n + 1)
    for i in range(1, n):
        alfa[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1]

    # Paso 3: resolver el sistema tridiagonal para c (método de Thomas)
    #         con c_0 = 0 y c_n = 0  (condiciones naturales)
    l = [1] * (n + 1)
    mu = [0] * (n + 1)
    z = [0] * (n + 1)
    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alfa[i] - h[i - 1] * z[i - 1]) / l[i]

    c = [0] * (n + 1)
    b = [0] * n
    d = [0] * n
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    a = y[:n]
    return SplineNatural(x, a, b, c[:n], d)


# ----------------------------------------------------------------------------
# Objeto spline: evaluar, mostrar, verificar y graficar
# ----------------------------------------------------------------------------

def _fmt(v):
    """Formato bonito: fracciones exactas o decimales con 6 cifras."""
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    return f"{v:.6g}"


class SplineNatural:
    def __init__(self, x, a, b, c, d):
        self.x = x                       # nodos x_0, ..., x_n
        self.a, self.b, self.c, self.d = a, b, c, d
        self.n = len(a)                  # número de tramos

    # -- en qué tramo cae un valor x ------------------------------------------
    def _tramo(self, t):
        if t < self.x[0] or t > self.x[-1]:
            raise ValueError(f"x = {t} está fuera del intervalo [{self.x[0]}, {self.x[-1]}].")
        for i in range(self.n):
            if t <= self.x[i + 1]:
                return i
        return self.n - 1

    # -- evaluación de S, S' y S'' ---------------------------------------------
    def __call__(self, t, derivada=0):
        """s(t) = S(t);  s(t, 1) = S'(t);  s(t, 2) = S''(t).  Acepta número o lista."""
        if isinstance(t, (list, tuple)):
            return [self(v, derivada) for v in t]
        i = self._tramo(t)
        return self.evaluar_tramo(i, t, derivada)

    def evaluar_tramo(self, i, t, derivada=0):
        """Evalúa el tramo S_i (o su derivada) en t, aunque t esté en el borde."""
        u = t - self.x[i]
        a, b, c, d = self.a[i], self.b[i], self.c[i], self.d[i]
        if derivada == 0:
            return a + b * u + c * u**2 + d * u**3
        if derivada == 1:
            return b + 2 * c * u + 3 * d * u**2
        if derivada == 2:
            return 2 * c + 6 * d * u
        if derivada == 3:
            return 6 * d
        return 0

    # -- impresión -------------------------------------------------------------
    def tramo_texto(self, i):
        xi = self.x[i]
        if xi == 0:
            base = "x"
        elif xi > 0:
            base = f"(x - {_fmt(xi)})"
        else:
            base = f"(x + {_fmt(-xi)})"
        partes = []
        for coef, termino in ((self.a[i], ""), (self.b[i], f"*{base}"),
                              (self.c[i], f"*{base}^2"), (self.d[i], f"*{base}^3")):
            if coef == 0:
                continue
            if not partes:
                partes.append(f"{'-' if coef < 0 else ''}{_fmt(abs(coef))}{termino}")
            else:
                partes.append(f" {'-' if coef < 0 else '+'} {_fmt(abs(coef))}{termino}")
        return "".join(partes) if partes else "0"

    def mostrar(self):
        print("Spline cúbico natural")
        print("S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3\n")
        encabezado = f"{'i':>3} | {'[x_i, x_i+1]':^16} | {'a_i':>10} {'b_i':>10} {'c_i':>10} {'d_i':>10}"
        print(encabezado)
        print("-" * len(encabezado))
        for i in range(self.n):
            intervalo = f"[{_fmt(self.x[i])}, {_fmt(self.x[i + 1])}]"
            print(f"{i:>3} | {intervalo:^16} | {_fmt(self.a[i]):>10} {_fmt(self.b[i]):>10} "
                  f"{_fmt(self.c[i]):>10} {_fmt(self.d[i]):>10}")
        print()
        for i in range(self.n):
            print(f"S_{i}(x) = {self.tramo_texto(i)},   "
                  f"{_fmt(self.x[i])} <= x <= {_fmt(self.x[i + 1])}")
        print()

    # -- verificación (mismos pasos que la guía) -------------------------------
    def verificar(self, y=None, tol=1e-9):
        """
        Comprueba: (1) interpolación, (2) continuidad de S, S', S'' en nodos
        interiores y (3) condiciones naturales S''(x_0) = S''(x_n) = 0.
        Si se entrega y, se compara con esos datos.
        """
        def igual(p, q):
            return p == q if isinstance(p - q, Fraction) else abs(p - q) < tol

        ok = True
        print("Verificación")
        print("1) Interpolación:")
        for i in range(self.n):
            izq = self.evaluar_tramo(i, self.x[i])
            der = self.evaluar_tramo(i, self.x[i + 1])
            print(f"   S_{i}({_fmt(self.x[i])}) = {_fmt(izq)},  S_{i}({_fmt(self.x[i + 1])}) = {_fmt(der)}")
            if y is not None:
                if not (igual(izq, _a_numero(y[i])) and igual(der, _a_numero(y[i + 1]))):
                    ok = False
                    print("      <-- no coincide con los datos")

        print("2) Empalmes en nodos interiores:")
        if self.n == 1:
            print("   (no hay nodos interiores)")
        nombres = ["S  ", "S' ", "S''"]
        for k in range(1, self.n):
            xk = self.x[k]
            for der in range(3):
                izq = self.evaluar_tramo(k - 1, xk, der)
                dch = self.evaluar_tramo(k, xk, der)
                marca = "OK" if igual(izq, dch) else "FALLA"
                ok = ok and marca == "OK"
                print(f"   x = {_fmt(xk)}: {nombres[der]} izq = {_fmt(izq):>10}, "
                      f"der = {_fmt(dch):>10}   {marca}")

        print("3) Condiciones naturales:")
        s0 = self.evaluar_tramo(0, self.x[0], 2)
        sn = self.evaluar_tramo(self.n - 1, self.x[-1], 2)
        for nombre, val in ((f"S''({_fmt(self.x[0])})", s0), (f"S''({_fmt(self.x[-1])})", sn)):
            marca = "OK" if igual(val, 0) else "FALLA"
            ok = ok and marca == "OK"
            print(f"   {nombre} = {_fmt(val)}   {marca}")

        print("Conclusión:", "SÍ es un spline cúbico natural." if ok else "NO cumple todas las condiciones.")
        print()
        return ok

    # -- gráfico ---------------------------------------------------------------
    def graficar(self, puntos_por_tramo=50, titulo="Spline cúbico natural"):
        import matplotlib.pyplot as plt
        for i in range(self.n):
            x0, x1 = float(self.x[i]), float(self.x[i + 1])
            ts = [x0 + (x1 - x0) * k / puntos_por_tramo for k in range(puntos_por_tramo + 1)]
            ys = [float(self.evaluar_tramo(i, t)) for t in ts]
            plt.plot(ts, ys, label=f"S_{i}")
        xs = [float(v) for v in self.x]
        ys = [float(self.evaluar_tramo(min(i, self.n - 1), self.x[i])) for i in range(self.n + 1)]
        plt.plot(xs, ys, "ko", label="datos")
        plt.title(titulo)
        plt.xlabel("x")
        plt.ylabel("S(x)")
        plt.grid(True)
        plt.legend()
        plt.show()


# ----------------------------------------------------------------------------
# Ejemplos de la guía
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Ejemplo 1 de la guía: datos (0, 0), (1, 1), (2, 0)")
    print("=" * 70)
    x = [0, 1, 2]
    y = [0, 1, 0]
    s = spline_natural(x, y)
    s.mostrar()
    s.verificar(y)
    print("S(0.5) =", _fmt(s(Fraction(1, 2))), "  (como decimal:", float(s(0.5)), ")\n")

    print("=" * 70)
    print("Ejercicio S1 de la guía: datos (-1, 1), (0, 0), (1, 1)")
    print("=" * 70)
    x = [-1, 0, 1]
    y = [1, 0, 1]
    s = spline_natural(x, y)
    s.mostrar()
    s.verificar(y)

    print("=" * 70)
    print("Ejemplo con decimales: f(x) = e^x en x = 0, 1, 2, 3")
    print("=" * 70)
    import math
    x = [0, 1, 2, 3]
    y = [math.exp(v) for v in x]
    s = spline_natural(x, y)
    s.mostrar()
    s.verificar(y)
