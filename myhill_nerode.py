"""
Teorema de Myhill-Nerode aplicado al AFD por el algoritmo de llenado de tabla
(distinguibilidad de estados).

Este programa REUTILIZA :
    shunting_yard.py    
    arbol_sintactico.py 
    afn_thompson.py     
    afd_subconjuntos.py 
    afd_minimizacion.py 

Teorema de Myhill-Nerode:
    Para un lenguaje L se define la relación de indistinguibilidad sobre las
    cadenas:  x ≡_L y  si y solo si  para toda cadena z,  xz ∈ L  <=>  yz ∈ L.
    L es regular si y solo si esta relación tiene un número FINITO de clases de
    equivalencia, y ese número (el índice) es exactamente la cantidad de estados
    del AFD mínimo (completo) que reconoce L.

Este módulo implementa el teorema con el algoritmo de llenado de tabla, que es
su forma algorítmica clásica:
  1. Se completa el AFD con un estado muerto para que sea total. Cada estado del
     AFD representa una clase de cadenas: las que llegan a ese estado.
  2. Tabla de distinguibilidad: para cada par de estados {p, q} se decide si son
     distinguibles.
       - base: si uno acepta y el otro no, son distinguibles; el sufijo testigo
         que los separa es ε (la cadena vacía).
       - paso: si {δ(p,a), δ(q,a)} ya es distinguible con testigo z para algún
         símbolo a, entonces {p, q} es distinguible con testigo a·z.
     Se repite hasta que la tabla no cambia.
  3. Los pares que quedan sin marcar son indistinguibles: se agrupan en clases de
     equivalencia. Esas clases son las clases de Myhill-Nerode y forman el AFD
     mínimo.
  4. Se reporta el índice, se verifica el teorema y se compara el resultado con
     la minimización por refinamiento de particiones (deben coincidir).

Uso:
    python myhill_nerode.py archivo_expresiones.txt "cadena_a_probar"
    python myhill_nerode.py archivo_expresiones.txt        (pide la cadena por teclado)

Requiere:
    pip install matplotlib
"""

import sys

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol
from afn_thompson import construir_afn
from afd_subconjuntos import construir_afd
from afd_minimizacion import (
    minimizar, EstadoAFDMin, AFDMin, dibujar_afd_min, _fmt_bloque,
)

# etiqueta del estado muerto que completa el AFD (destino de las transiciones
# no definidas); representa la clase de las cadenas sin extensión posible a L
MUERTO = "muerto"
EPSILON = "ε"


def _delta(estado, simbolo):
    """Transición total: si no existe, cae al estado muerto."""
    if estado is MUERTO:
        return MUERTO
    return estado.transiciones.get(simbolo, MUERTO)


def _nombre(estado):
    return MUERTO if estado is MUERTO else estado.nombre


# 1) Algoritmo de llenado de tabla (distinguibilidad)
def construir_tabla(afd, pasos):
    """
    Aplica el llenado de tabla sobre 'afd' completado con estado muerto.
    Retorna:
        estados       lista de estados (incluye MUERTO si es alcanzable)
        distinguible  dict {frozenset({p,q}): sufijo_testigo}   (pares distinguibles)
        hay_muerto    True si el estado muerto es alcanzable (existe clase muerta)
    """
    alfabeto = afd.alfabeto
    aceptacion = set(afd.aceptacion)

    # el estado muerto solo es una clase real si alguna transición faltaba
    hay_muerto = any(
        s not in e.transiciones for e in afd.estados for s in alfabeto
    )
    estados = list(afd.estados) + ([MUERTO] if hay_muerto else [])

    def acepta(estado):
        return estado is not MUERTO and estado in aceptacion

    # todos los pares no ordenados {p, q} con p != q
    pares = []
    for i in range(len(estados)):
        for j in range(i + 1, len(estados)):
            pares.append((estados[i], estados[j]))

    distinguible = {}   # frozenset({p,q}) -> sufijo testigo (str)

    # base: aceptación vs no aceptación, testigo ε
    for p, q in pares:
        if acepta(p) != acepta(q):
            distinguible[frozenset((p, q))] = ""   # "" representa ε
    pasos.append(
        f"  Base: {len(distinguible)} par(es) marcados por diferir en aceptación "
        f"(testigo ε)."
    )

    #  paso: propagar hasta punto fijo
    ronda = 0
    while True:
        ronda += 1
        nuevos = 0
        for p, q in pares:
            clave = frozenset((p, q))
            if clave in distinguible:
                continue
            for a in alfabeto:
                dp, dq = _delta(p, a), _delta(q, a)
                if dp is dq:
                    continue
                sub = distinguible.get(frozenset((dp, dq)))
                if sub is not None:
                    distinguible[clave] = a + sub   # testigo a·z
                    nuevos += 1
                    break
        if nuevos == 0:
            pasos.append(f"  Ronda {ronda}: 0 pares nuevos, la tabla queda fija.")
            break
        pasos.append(f"  Ronda {ronda}: {nuevos} par(es) nuevos marcados.")

    return estados, distinguible, hay_muerto


# 2) Clases de equivalencia (unión de pares indistinguibles)
def clases_de_equivalencia(estados, distinguible):
    """Agrupa los estados en clases: dos estados van juntos si NO son distinguibles."""
    representante = {e: e for e in estados}

    def raiz(e):
        while representante[e] is not e:
            representante[e] = representante[representante[e]]
            e = representante[e]
        return e

    def unir(a, b):
        representante[raiz(a)] = raiz(b)

    for i in range(len(estados)):
        for j in range(i + 1, len(estados)):
            p, q = estados[i], estados[j]
            if frozenset((p, q)) not in distinguible:
                unir(p, q)

    grupos = {}
    for e in estados:
        grupos.setdefault(raiz(e), []).append(e)
    return [frozenset(g) for g in grupos.values()]


# 3) Cadena representante de cada clase (BFS de menor longitud desde el inicio)
def cadenas_representantes(afd, estados):
    """Cadena de menor longitud que lleva del inicio a cada estado (incluye muerto)."""
    alfabeto = afd.alfabeto
    representante = {afd.inicio: ""}
    cola = [afd.inicio]
    while cola:
        actual = cola.pop(0)
        for a in alfabeto:
            destino = _delta(actual, a)
            if destino not in representante:
                representante[destino] = representante[actual] + a
                if destino is not MUERTO:
                    cola.append(destino)
    return representante


# 4) AFD mínimo construido con las clases de Myhill-Nerode (para dibujar/simular)
def afd_desde_clases(afd, clases, hay_muerto):
    """
    Construye un AFDMin a partir de las clases de Myhill-Nerode, descartando la
    clase muerta (igual que el AFD mínimo parcial). Reutiliza EstadoAFDMin/AFDMin.
    """
    EstadoAFDMin._contador = 0
    aceptacion = set(afd.aceptacion)

    # localizar la clase muerta
    clase_muerta = None
    if hay_muerto:
        for clase in clases:
            if any(e is MUERTO for e in clase):
                clase_muerta = clase
                break

    utiles = [c for c in clases if c != clase_muerta]
    # la clase del estado inicial va primero -> queda como M0
    utiles.sort(key=lambda c: (afd.inicio not in c))

    estado_de_clase = {}
    estados_min = []
    clase_de_estado = {}
    for clase in clases:
        for e in clase:
            clase_de_estado[e] = clase

    for clase in utiles:
        bloque = frozenset(e for e in clase if e is not MUERTO)
        nuevo = EstadoAFDMin(bloque)
        estado_de_clase[clase] = nuevo
        estados_min.append(nuevo)

    for clase in utiles:
        origen = estado_de_clase[clase]
        rep = next(e for e in clase if e is not MUERTO)
        for a in afd.alfabeto:
            destino = _delta(rep, a)
            clase_destino = clase_de_estado[destino]
            if clase_destino == clase_muerta:
                continue
            origen.transiciones[a] = estado_de_clase[clase_destino]

    inicio = estado_de_clase[clase_de_estado[afd.inicio]]
    aceptacion_min = {
        estado_de_clase[c] for c in utiles if any(e in aceptacion for e in c)
    }
    return AFDMin(inicio, estados_min, aceptacion_min, afd.alfabeto)


# Impresión de la tabla de distinguibilidad (triangular)
def imprimir_tabla(estados, distinguible):
    etiquetas = [_nombre(e) for e in estados]
    ancho = max(len(x) for x in etiquetas) + 1
    # cada celda: 'x' si indistinguible, o el testigo (ε para vacío)
    print("  Tabla de distinguibilidad (testigo que separa cada par; x = indistinguibles):")
    encabezado = " " * (ancho + 2) + "".join(f"{etiquetas[j]:>{ancho+2}}" for j in range(len(estados) - 1))
    print("  " + encabezado)
    for i in range(1, len(estados)):
        fila = f"{etiquetas[i]:>{ancho}}  "
        for j in range(i):
            clave = frozenset((estados[i], estados[j]))
            if clave in distinguible:
                testigo = distinguible[clave]
                celda = EPSILON if testigo == "" else testigo
            else:
                celda = "x"
            fila += f"{celda:>{ancho+2}}"
        print("  " + fila)


# Procesamiento del archivo
def procesar_archivo(ruta_archivo, cadena_w):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            lineas = [linea.rstrip("\n") for linea in archivo]
    except FileNotFoundError:
        print(f"ERROR: no se encontró el archivo '{ruta_archivo}'.")
        sys.exit(1)

    print(f"Archivo a procesar: {ruta_archivo}")
    print(f"Cadena w a evaluar: '{cadena_w}'")
    print("=" * 72)

    numero = 0
    for numero_linea, expresion in enumerate(lineas, start=1):
        if expresion.strip() == "":
            continue
        numero += 1

        print(f"\nLínea {numero_linea} — expresión regular r: {expresion}")
        print("-" * 72)

        # pipeline hasta el AFD
        try:
            postfix_tokens, _ = a_postfix(expresion)
            raiz = construir_arbol(postfix_tokens, [])
            afn = construir_afn(raiz, [])
            afd = construir_afd(afn, [])
        except ErrorExpresion as error:
            print(f"  ERROR en el pipeline: {error}")
            print("=" * 72)
            continue
        print(f"AFD por subconjuntos: {len(afd.estados)} estados "
              f"(D0 ... D{len(afd.estados) - 1})")

        # 1) llenado de tabla
        print("\nLlenado de tabla (distinguibilidad de estados), pasos:")
        pasos_tabla = []
        estados, distinguible, hay_muerto = construir_tabla(afd, pasos_tabla)
        for paso in pasos_tabla:
            print(paso)
        if hay_muerto:
            print("  (El AFD se completó con un estado 'muerto': existen cadenas "
                  "sin extensión posible a L.)")
        else:
            print("  (El AFD ya era total: todo prefijo puede extenderse a L, "
                  "no hay clase muerta.)")

        print()
        imprimir_tabla(estados, distinguible)

        # 2) clases de equivalencia
        clases = clases_de_equivalencia(estados, distinguible)
        reps = cadenas_representantes(afd, estados)

        def rep_de_clase(clase):
            cadena = min((reps[e] for e in clase), key=len)
            return EPSILON if cadena == "" else cadena

        aceptacion = set(afd.aceptacion)
        print(f"\nClases de equivalencia de Myhill-Nerode ({len(clases)}):")
        # ordenar: primero la del inicio, luego por representante
        clases_ordenadas = sorted(
            clases, key=lambda c: (afd.inicio not in c, len(rep_de_clase(c)), rep_de_clase(c))
        )
        for indice, clase in enumerate(clases_ordenadas):
            miembros = "{" + ", ".join(sorted(_nombre(e) for e in clase)) + "}"
            es_muerta = any(e is MUERTO for e in clase)
            if es_muerta:
                tipo = "muerta (sin extensión a L)"
            elif any(e in aceptacion for e in clase):
                tipo = "de aceptación"
            else:
                tipo = "de rechazo"
            marca_inicio = "  <- clase del inicio" if afd.inicio in clase else ""
            print(f"  C{indice}: representante '{rep_de_clase(clase)}'  "
                  f"estados {miembros}  ({tipo}){marca_inicio}")

        # 3) índice y verificación del teorema
        indice = len(clases)
        afd_min = minimizar(afd, [])
        clases_vivas = indice - (1 if hay_muerto else 0)

        print(f"\nÍndice de Myhill-Nerode (clases de equivalencia de Σ*): {indice}")
        print("El índice es finito, por lo tanto, por el teorema de Myhill-Nerode, "
              "L(r) es un lenguaje regular.")
        print(f"  Clases 'vivas' (sin la muerta)     : {clases_vivas}")
        print(f"  Estados del AFD mínimo parcial     : {len(afd_min.estados)} "
              f"(por refinamiento de particiones)")
        coincide = clases_vivas == len(afd_min.estados)
        print(f"  Verificación (deben coincidir)     : "
              f"{'OK, coinciden' if coincide else 'NO COINCIDEN'}")
        if hay_muerto:
            print(f"  Nota: el índice ({indice}) = estados del mínimo parcial "
                  f"({len(afd_min.estados)}) + 1 clase muerta.")

        # 4) autómata de las clases: dibujo y simulación de w
        afd_mn = afd_desde_clases(afd, clases, hay_muerto)
        ruta_imagen = f"myhill_{numero}.png"
        dibujar_afd_min(afd_mn, f"AFD mínimo (Myhill-Nerode) para: {expresion}", ruta_imagen)
        print(f"\n  Grafo del AFD mínimo por Myhill-Nerode guardado en: {ruta_imagen}")

        print(f"\nSimulación con w = '{cadena_w}' sobre el autómata de las clases:")
        pasos_sim = []
        aceptada = afd_mn.acepta(cadena_w, pasos_sim)
        for paso in pasos_sim:
            print(paso)
        print(f"\n  ¿w ∈ L(r)?  ->  {'sí' if aceptada else 'no'}")
        print("=" * 72)


def main():
    if len(sys.argv) == 3:
        ruta_archivo = sys.argv[1]
        cadena_w = sys.argv[2]
    elif len(sys.argv) == 2:
        ruta_archivo = sys.argv[1]
        cadena_w = input("Ingrese la cadena w a evaluar: ")
    else:
        print('Uso: python myhill_nerode.py <archivo.txt> ["cadena_w"]')
        sys.exit(1)

    procesar_archivo(ruta_archivo, cadena_w)


if __name__ == "__main__":
    main()
