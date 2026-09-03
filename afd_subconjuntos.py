"""
AFD por construcción de subconjuntos, a partir del AFN de Thompson.

Este programa REUTILIZA :
    shunting_yard.py    
    arbol_sintactico.py 
    afn_thompson.py      

y agrega:
  1. Construcción del AFD aplicando el algoritmo de construcción de
     subconjuntos (subset construction) sobre el AFN:
       - el estado inicial del AFD es la cerradura-ε del estado inicial del AFN
       - por cada estado del AFD y cada símbolo del alfabeto, se calcula
         mover + cerradura-ε para obtener el siguiente estado del AFD
       - un estado del AFD es de aceptación si el conjunto de estados del
         AFN que representa contiene al estado de aceptación del AFN
  2. Dibujo del grafo del AFD (estado inicial, estados adicionales, estado
     de aceptación, transiciones con su símbolo -- ya no hay transiciones ε
     porque el AFD es determinista).
  3. Simulación del AFD sobre una cadena w: se sigue una única transición
     por símbolo (sin cerradura-ε, sin no-determinismo) e indica si
     w pertenece a L(r) con "sí" / "no".

Uso:
    python afd_subconjuntos.py archivo_expresiones.txt "cadena_a_probar"
    python afd_subconjuntos.py archivo_expresiones.txt       


"""

import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dibujo import dibujar_automata

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol, imprimir_arbol
from afn_thompson import construir_afn, _fmt


# Objetos del AFD
class EstadoAFD:
    """
    Un estado del AFD. 'nombre' es el identificador visible (Dn) y
    'conjunto_afn' es el frozenset de estados del AFN que representa.
    """

    _contador = 0

    def __init__(self, conjunto_afn):
        self.nombre = f"D{EstadoAFD._contador}"
        EstadoAFD._contador += 1
        self.conjunto_afn = conjunto_afn        # frozenset de Estado (del AFN)
        self.transiciones = {}                  # simbolo -> EstadoAFD

    def __repr__(self):
        return self.nombre


class AFD:
    """AFD completo, con su estado inicial, los estados de aceptación y el alfabeto."""

    def __init__(self, inicio, estados, aceptacion, alfabeto):
        self.inicio = inicio
        self.estados = estados              # lista de EstadoAFD, en orden de creación
        self.aceptacion = aceptacion        # set de EstadoAFD que son de aceptación
        self.alfabeto = alfabeto

    def acepta(self, cadena, pasos=None):
        actual = self.inicio
        if pasos is not None:
            pasos.append(f"  Inicio            -> estado: {actual}  {_fmt(actual.conjunto_afn)}")

        for i, caracter in enumerate(cadena):
            siguiente = actual.transiciones.get(caracter)
            if siguiente is None:
                if pasos is not None:
                    pasos.append(
                        f"  Leyendo '{caracter}' (pos {i}) -> no existe transición "
                        f"desde {actual}: la cadena se rechaza aquí."
                    )
                return False
            actual = siguiente
            if pasos is not None:
                pasos.append(
                    f"  Leyendo '{caracter}' (pos {i}) -> estado: {actual}  "
                    f"{_fmt(actual.conjunto_afn)}"
                )

        aceptada = actual in self.aceptacion
        if pasos is not None:
            pasos.append(
                f"  Fin de la cadena. ¿{actual} es estado de aceptación? -> "
                f"{'sí' if aceptada else 'no'}"
            )
        return aceptada


# Construcción de subconjuntos
def construir_afd(afn, pasos):
    """
    Aplica el algoritmo de construcción de subconjuntos sobre 'afn'.
    Retorna un objeto AFD. 'pasos' recibe la traza en texto de cada
    estado nuevo creado y cada transición calculada.
    """
    EstadoAFD._contador = 0
    alfabeto = afn.alfabeto

    cerradura_inicial = frozenset(afn.cerradura_epsilon({afn.inicio}))
    inicio = EstadoAFD(cerradura_inicial)
    pasos.append(f"  Estado inicial {inicio} = cerradura-ε(inicio AFN) = {_fmt(cerradura_inicial)}")

    estados_por_conjunto = {cerradura_inicial: inicio}
    todos_los_estados = [inicio]
    pendientes = [inicio]

    while pendientes:
        estado_actual = pendientes.pop(0)
        for simbolo in alfabeto:
            movidos = afn.mover(estado_actual.conjunto_afn, simbolo)
            si_vacio = len(movidos) == 0
            if si_vacio:
                pasos.append(
                    f"  {estado_actual} --{simbolo}--> (conjunto vacío, no se crea transición)"
                )
                continue

            nuevo_conjunto = frozenset(afn.cerradura_epsilon(movidos))

            if nuevo_conjunto in estados_por_conjunto:
                estado_destino = estados_por_conjunto[nuevo_conjunto]
                pasos.append(
                    f"  {estado_actual} --{simbolo}--> {estado_destino} "
                    f"(ya existía) = {_fmt(nuevo_conjunto)}"
                )
            else:
                estado_destino = EstadoAFD(nuevo_conjunto)
                estados_por_conjunto[nuevo_conjunto] = estado_destino
                todos_los_estados.append(estado_destino)
                pendientes.append(estado_destino)
                pasos.append(
                    f"  {estado_actual} --{simbolo}--> {estado_destino} "
                    f"(estado nuevo) = {_fmt(nuevo_conjunto)}"
                )

            estado_actual.transiciones[simbolo] = estado_destino

    aceptacion = {
        estado for estado in todos_los_estados
        if afn.aceptacion in estado.conjunto_afn
    }

    return AFD(inicio, todos_los_estados, aceptacion, alfabeto)


# Dibujo del grafo del AFD
def _offsets_bucle(angulo_grados):
    """
    Calcula los tres puntos que arman un auto-bucle (flecha de un estado hacia
    si mismo), rotados 'angulo_grados' alrededor del estado. Asi, cuando un
    estado tiene varios simbolos en auto-bucle (por ejemplo 'a' y 'b' sobre el
    mismo estado), cada uno queda en un angulo distinto y no se superponen.
    Retorna (punta_flecha, inicio_flecha, posicion_etiqueta), cada uno como
    un desplazamiento (dx, dy) relativo al centro del estado.
    """
    theta = math.radians(angulo_grados)

    def rotar(dx, dy):
        return (dx * math.cos(theta) - dy * math.sin(theta),
                dx * math.sin(theta) + dy * math.cos(theta))

    punta = rotar(0.0, 0.30)
    inicio = rotar(0.42, 0.66)
    etiqueta = rotar(0.50, 0.86)
    return punta, inicio, etiqueta


def _curvatura(x1, y1, x2, y2):
    """
    Lado hacia el que se curva una transicion. Cuando los dos estados estan al
    mismo nivel vertical se usa la posicion horizontal para decidir, de modo que
    la flecha de ida y la de vuelta no queden exactamente encimadas.
    """
    if y1 != y2:
        return 0.18 if y1 < y2 else -0.18
    return 0.18 if x1 <= x2 else -0.18


def _ajustar_ejes(ejes):
    """
    Deja los ejes con proporcion 1:1 (para que los estados se vean como
    circulos y no como ovalos aplastados) y garantiza un area minima, que es
    lo que se necesita cuando el automata tiene un solo estado.
    """
    ejes.set_aspect("equal", adjustable="datalim")
    ejes.relim()
    ejes.autoscale_view()

    x0, x1 = ejes.get_xlim()
    y0, y1 = ejes.get_ylim()

    # area minima alrededor del contenido
    ancho_minimo, alto_minimo = 4.6, 3.2
    if x1 - x0 < ancho_minimo:
        centro = (x0 + x1) / 2
        x0, x1 = centro - ancho_minimo / 2, centro + ancho_minimo / 2
    if y1 - y0 < alto_minimo:
        centro = (y0 + y1) / 2
        y0, y1 = centro - alto_minimo / 2, centro + alto_minimo / 2

    ejes.set_xlim(x0 - 0.45, x1 + 0.45)
    ejes.set_ylim(y0 - 0.45, y1 + 0.75)


def dibujar_afd(afd, titulo, ruta_imagen):
    """
    Dibuja el AFD por niveles (BFS desde el inicio). El dibujo en sí lo hace
    el módulo dibujo.py, que es el mismo que usan el AFN y el AFD mínimo.
    """
    nivel = {afd.inicio: 0}
    cola = [afd.inicio]
    while cola:
        estado = cola.pop(0)
        for destino in estado.transiciones.values():
            if destino not in nivel:
                nivel[destino] = nivel[estado] + 1
                cola.append(destino)
    for estado in afd.estados:
        nivel.setdefault(estado, 0)

    por_nivel = {}
    for estado in afd.estados:
        por_nivel.setdefault(nivel[estado], []).append(estado)

    posiciones = {}
    for nivel_actual, estados in por_nivel.items():
        estados.sort(key=lambda e: e.nombre)
        for indice, estado in enumerate(estados):
            desplazamiento = (len(estados) - 1) / 2
            posiciones[estado] = (nivel_actual * 2.4, (indice - desplazamiento) * 1.7)

    transiciones = [
        (estado, simbolo, destino)
        for estado in afd.estados
        for simbolo, destino in estado.transiciones.items()
    ]
    nombres = {estado: estado.nombre for estado in afd.estados}

    dibujar_automata(titulo, ruta_imagen, posiciones, transiciones,
                     afd.inicio, afd.aceptacion, nombres)


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

        # 1) infix -> postfix
        try:
            postfix_tokens, _ = a_postfix(expresion)
        except ErrorExpresion as error:
            print(f"  ERROR en Shunting Yard: {error}")
            print("=" * 72)
            continue
        print(f"Postfix: {' '.join(str(t) for t in postfix_tokens)}")

        # 2) postfix -> árbol sintáctico
        try:
            raiz = construir_arbol(postfix_tokens, [])
        except ErrorExpresion as error:
            print(f"  ERROR construyendo el árbol: {error}")
            print("=" * 72)
            continue

        # 3) árbol -> AFN (Thompson)
        try:
            afn = construir_afn(raiz, [])
        except ErrorExpresion as error:
            print(f"  ERROR construyendo el AFN: {error}")
            print("=" * 72)
            continue
        print(f"AFN: {len(afn.estados)} estados, alfabeto {{{', '.join(afn.alfabeto)}}}")

        # 4) AFN -> AFD (construcción de subconjuntos)
        print("\nConstrucción de subconjuntos (AFN -> AFD), pasos:")
        pasos_afd = []
        afd = construir_afd(afn, pasos_afd)
        for paso in pasos_afd:
            print(paso)

        print(f"\nAFD generado:")
        print(f"  Estado inicial      : {afd.inicio}")
        print(f"  Estados de aceptación: {', '.join(str(e) for e in sorted(afd.aceptacion, key=lambda x: x.nombre)) or '(ninguno)'}")
        print(f"  Total de estados    : {len(afd.estados)}")
        print(f"  Alfabeto            : {{{', '.join(afd.alfabeto)}}}")
        print("  Transiciones:")
        for estado in afd.estados:
            for simbolo, destino in estado.transiciones.items():
                print(f"    {estado} --{simbolo}--> {destino}")

        # 5) dibujo
        ruta_imagen = f"afd_{numero}.png"
        dibujar_afd(afd, f"AFD (subconjuntos) para: {expresion}", ruta_imagen)
        print(f"\n  Grafo del AFD guardado en: {ruta_imagen}")

        # 6) simulación
        print(f"\nSimulación del AFD con w = '{cadena_w}':")
        pasos_sim = []
        aceptada = afd.acepta(cadena_w, pasos_sim)
        for paso in pasos_sim:
            print(paso)

        respuesta = "sí" if aceptada else "no"
        print(f"\n  ¿w ∈ L(r)?  ->  {respuesta}")
        print("=" * 72)


def main():
    if len(sys.argv) == 3:
        ruta_archivo = sys.argv[1]
        cadena_w = sys.argv[2]
    elif len(sys.argv) == 2:
        ruta_archivo = sys.argv[1]
        cadena_w = input("Ingrese la cadena w a evaluar: ")
    else:
        print('Uso: python afd_subconjuntos.py <archivo.txt> ["cadena_w"]')
        sys.exit(1)

    procesar_archivo(ruta_archivo, cadena_w)


if __name__ == "__main__":
    main()
