"""
Minimización de un AFD por el algoritmo de refinamiento de particiones
(equivalente al método de llenado de tabla / algoritmo de Moore).

Este programa REUTILIZA los laboratorios anteriores:
    shunting_yard.py    -> infix a postfix
    arbol_sintactico.py -> postfix a árbol sintáctico
    afn_thompson.py     -> árbol sintáctico a AFN (Thompson)
    afd_subconjuntos.py -> AFN a AFD (construcción de subconjuntos)

y agrega:
  1. Minimización del AFD:
       - se completa el AFD con un estado trampa implícito para que todas
         las transiciones estén definidas (AFD total).
       - partición inicial en dos grupos: estados de aceptación y estados
         de no aceptación.
       - se refina cada grupo: dos estados quedan juntos solo si, para cada
         símbolo del alfabeto, van a parar al mismo grupo. Se repite hasta
         que la partición ya no cambia.
       - cada grupo (bloque) de la partición final se vuelve un estado del
         AFD mínimo. El estado trampa (bloque muerto, no aceptación, que solo
         va hacia sí mismo) se descarta para mostrar el AFD mínimo parcial.
  2. Dibujo del grafo del AFD mínimo (mismo estilo que el AFN/AFD).
  3. Simulación del AFD mínimo sobre una cadena w: una sola transición por
     símbolo; indica si w pertenece a L(r) con "sí" / "no".

Uso:
    python afd_minimizacion.py archivo_expresiones.txt "cadena_a_probar"
    python afd_minimizacion.py archivo_expresiones.txt        (pide la cadena por teclado)

Requiere:
    pip install matplotlib
"""

import math
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dibujo import dibujar_automata

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol
from afn_thompson import construir_afn
from afd_subconjuntos import construir_afd


# ---------------------------------------------------------------------------
# Objetos del AFD mínimo
# ---------------------------------------------------------------------------
class EstadoAFDMin:
    """
    Un estado del AFD mínimo. 'nombre' es el identificador visible (Mn) y
    'bloque' es el frozenset de EstadoAFD (del AFD por subconjuntos) que
    se fusionan en este estado.
    """

    _contador = 0

    def __init__(self, bloque):
        self.nombre = f"M{EstadoAFDMin._contador}"
        EstadoAFDMin._contador += 1
        self.bloque = bloque            # frozenset de EstadoAFD
        self.transiciones = {}          # simbolo -> EstadoAFDMin

    def __repr__(self):
        return self.nombre


class AFDMin:
    """AFD mínimo, con su estado inicial, los de aceptación y el alfabeto."""

    def __init__(self, inicio, estados, aceptacion, alfabeto):
        self.inicio = inicio
        self.estados = estados          # lista de EstadoAFDMin
        self.aceptacion = aceptacion    # set de EstadoAFDMin de aceptación
        self.alfabeto = alfabeto

    def acepta(self, cadena, pasos=None):
        actual = self.inicio
        if pasos is not None:
            pasos.append(f"  Inicio            -> estado: {actual}  {_fmt_bloque(actual.bloque)}")

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
                    f"{_fmt_bloque(actual.bloque)}"
                )

        aceptada = actual in self.aceptacion
        if pasos is not None:
            pasos.append(
                f"  Fin de la cadena. ¿{actual} es estado de aceptación? -> "
                f"{'sí' if aceptada else 'no'}"
            )
        return aceptada


def _fmt_bloque(bloque):
    """Muestra un bloque como el conjunto de estados del AFD que lo forman."""
    return "{" + ", ".join(e.nombre for e in sorted(bloque, key=lambda x: x.nombre)) + "}"


# ---------------------------------------------------------------------------
# Minimización por refinamiento de particiones
# ---------------------------------------------------------------------------
# centinela para el estado trampa (destino de las transiciones no definidas)
_TRAMPA = "TRAMPA"


def _delta(estado, simbolo):
    """Transición total: si no existe, cae al estado trampa."""
    if estado is _TRAMPA:
        return _TRAMPA
    return estado.transiciones.get(simbolo, _TRAMPA)


def minimizar(afd, pasos):
    """
    Minimiza 'afd' (un objeto AFD de afd_subconjuntos) por refinamiento de
    particiones. Retorna un AFDMin. 'pasos' recibe la traza en texto de la
    partición inicial y de cada ronda de refinamiento.
    """
    EstadoAFDMin._contador = 0
    alfabeto = afd.alfabeto

    # todos los estados, incluyendo el trampa que completa el AFD
    estados = list(afd.estados) + [_TRAMPA]
    aceptacion = set(afd.aceptacion)

    def es_aceptacion(estado):
        return estado is not _TRAMPA and estado in aceptacion

    # ---- partición inicial: aceptación vs no aceptación
    finales = frozenset(e for e in estados if es_aceptacion(e))
    no_finales = frozenset(e for e in estados if not es_aceptacion(e))
    particion = [b for b in (finales, no_finales) if b]

    pasos.append("  Partición inicial (aceptación vs no aceptación):")
    for i, bloque in enumerate(particion):
        pasos.append(f"    G{i} = {_fmt_bloque_estados(bloque)}")

    # ---- refinamiento
    ronda = 0
    while True:
        ronda += 1
        indice = {}
        for i, bloque in enumerate(particion):
            for estado in bloque:
                indice[estado] = i

        nueva_particion = []
        for bloque in particion:
            # subdividir el bloque por su "firma": a qué bloque va cada símbolo
            grupos = {}
            for estado in bloque:
                firma = tuple(indice[_delta(estado, s)] for s in alfabeto)
                grupos.setdefault(firma, []).append(estado)
            for grupo in grupos.values():
                nueva_particion.append(frozenset(grupo))

        if len(nueva_particion) == len(particion):
            pasos.append(f"  Ronda {ronda}: la partición no cambió, el refinamiento termina.")
            break

        particion = nueva_particion
        pasos.append(f"  Ronda {ronda}: se refina en {len(particion)} grupos:")
        for i, bloque in enumerate(particion):
            pasos.append(f"    G{i} = {_fmt_bloque_estados(bloque)}")

    # ---- identificar el bloque muerto (el que contiene el estado trampa)
    bloque_muerto = None
    for bloque in particion:
        if _TRAMPA in bloque:
            bloque_muerto = bloque
            break

    # el bloque muerto solo se descarta si no contiene al estado inicial
    # (si lo contuviera, el lenguaje sería vacío y hay que conservarlo)
    inicio_en_muerto = bloque_muerto is not None and afd.inicio in bloque_muerto
    descartar_muerto = bloque_muerto is not None and not inicio_en_muerto

    # ---- construir los estados del AFD mínimo
    # el bloque del estado inicial va primero para que quede como M0
    bloques_utiles = [b for b in particion if not (descartar_muerto and b is bloque_muerto)]
    bloques_utiles.sort(key=lambda b: (afd.inicio not in b))  # inicio primero

    # cada bloque -> EstadoAFDMin (con el trampa retirado del bloque)
    estado_de_bloque = {}
    estados_min = []
    for bloque in bloques_utiles:
        bloque_limpio = frozenset(e for e in bloque if e is not _TRAMPA)
        nuevo = EstadoAFDMin(bloque_limpio)
        estado_de_bloque[bloque] = nuevo
        estados_min.append(nuevo)

    # mapa: estado del AFD -> su bloque en la partición
    bloque_de_estado = {}
    for bloque in particion:
        for estado in bloque:
            bloque_de_estado[estado] = bloque

    # transiciones: se toma un representante de cada bloque
    for bloque in bloques_utiles:
        origen = estado_de_bloque[bloque]
        representante = next(e for e in bloque if e is not _TRAMPA)
        for simbolo in alfabeto:
            destino = _delta(representante, simbolo)
            bloque_destino = bloque_de_estado[destino]
            # si va al bloque muerto descartado, no se crea la transición
            if descartar_muerto and bloque_destino is bloque_muerto:
                continue
            origen.transiciones[simbolo] = estado_de_bloque[bloque_destino]

    inicio = estado_de_bloque[bloque_de_estado[afd.inicio]]
    aceptacion_min = {
        estado_de_bloque[b] for b in bloques_utiles
        if any(es_aceptacion(e) for e in b)
    }

    return AFDMin(inicio, estados_min, aceptacion_min, alfabeto)


def _fmt_bloque_estados(bloque):
    """Como _fmt_bloque pero tolera el centinela del estado trampa."""
    nombres = []
    for e in bloque:
        nombres.append("trampa" if e is _TRAMPA else e.nombre)
    return "{" + ", ".join(sorted(nombres)) + "}"


def _offsets_bucle(angulo_grados):
    """
    Calcula los tres puntos que arma un auto-bucle (flecha de un estado hacia
    sí mismo), rotados 'angulo_grados' alrededor del estado. Así, cuando un
    estado tiene varios símbolos en auto-bucle, cada uno queda en un ángulo
    distinto y no se superponen entre sí.
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
    Lado hacia el que se curva una transición. Cuando los dos estados están al
    mismo nivel vertical se usa la posición horizontal para decidir, de modo que
    la flecha de ida y la de vuelta no queden exactamente encimadas.
    """
    if y1 != y2:
        return 0.18 if y1 < y2 else -0.18
    return 0.18 if x1 <= x2 else -0.18


def _ajustar_ejes(ejes):
    """
    Deja los ejes con proporción 1:1 (para que los estados se vean como
    círculos y no como óvalos aplastados) y garantiza un área mínima, que es
    lo que se necesita cuando el autómata tiene un solo estado.
    """
    ejes.set_aspect("equal", adjustable="datalim")
    ejes.relim()
    ejes.autoscale_view()

    x0, x1 = ejes.get_xlim()
    y0, y1 = ejes.get_ylim()

    ancho_minimo, alto_minimo = 4.6, 3.2
    if x1 - x0 < ancho_minimo:
        centro = (x0 + x1) / 2
        x0, x1 = centro - ancho_minimo / 2, centro + ancho_minimo / 2
    if y1 - y0 < alto_minimo:
        centro = (y0 + y1) / 2
        y0, y1 = centro - alto_minimo / 2, centro + alto_minimo / 2

    ejes.set_xlim(x0 - 0.45, x1 + 0.45)
    ejes.set_ylim(y0 - 0.45, y1 + 0.75)


# ---------------------------------------------------------------------------
# Dibujo del grafo del AFD mínimo
# ---------------------------------------------------------------------------
def dibujar_afd_min(afd, titulo, ruta_imagen):
    """
    Dibuja el AFD mínimo por niveles (BFS desde el inicio), usando el mismo
    módulo de dibujo que el AFN y el AFD por subconjuntos.
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


# ---------------------------------------------------------------------------
# Procesamiento del archivo
# ---------------------------------------------------------------------------
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

        # 1) infix -> postfix -> árbol -> AFN -> AFD
        try:
            postfix_tokens, _ = a_postfix(expresion)
            raiz = construir_arbol(postfix_tokens, [])
            afn = construir_afn(raiz, [])
            afd = construir_afd(afn, [])
        except ErrorExpresion as error:
            print(f"  ERROR en el pipeline: {error}")
            print("=" * 72)
            continue
        print(f"AFD por subconjuntos: {len(afd.estados)} estados")

        # 2) AFD -> AFD mínimo
        print("\nMinimización (refinamiento de particiones), pasos:")
        pasos_min = []
        afd_min = minimizar(afd, pasos_min)
        for paso in pasos_min:
            print(paso)

        print(f"\nAFD mínimo generado:")
        print(f"  Estado inicial      : {afd_min.inicio}")
        print(f"  Estados de aceptación: "
              f"{', '.join(str(e) for e in sorted(afd_min.aceptacion, key=lambda x: x.nombre)) or '(ninguno)'}")
        print(f"  Total de estados    : {len(afd_min.estados)}  "
              f"(el AFD por subconjuntos tenía {len(afd.estados)})")
        print(f"  Alfabeto            : {{{', '.join(afd_min.alfabeto)}}}")
        print("  Transiciones:")
        for estado in afd_min.estados:
            for simbolo, destino in estado.transiciones.items():
                print(f"    {estado} --{simbolo}--> {destino}   {_fmt_bloque(estado.bloque)}")

        # 3) dibujo
        ruta_imagen = f"afd_min_{numero}.png"
        dibujar_afd_min(afd_min, f"AFD mínimo para: {expresion}", ruta_imagen)
        print(f"\n  Grafo del AFD mínimo guardado en: {ruta_imagen}")

        # 4) simulación
        print(f"\nSimulación del AFD mínimo con w = '{cadena_w}':")
        pasos_sim = []
        aceptada = afd_min.acepta(cadena_w, pasos_sim)
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
        print('Uso: python afd_minimizacion.py <archivo.txt> ["cadena_w"]')
        sys.exit(1)

    procesar_archivo(ruta_archivo, cadena_w)


if __name__ == "__main__":
    main()
