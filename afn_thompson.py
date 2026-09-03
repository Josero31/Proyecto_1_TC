"""
AFN por construcción de Thompson a partir del árbol sintáctico de una regex.

Este programa REUTILIZA:
    shunting_yard.py    
    arbol_sintactico.py 

y agrega:
  1. Construcción del AFN aplicando el algoritmo de Thompson recorriendo el
     árbol sintáctico de abajo hacia arriba. Cada nodo produce un fragmento
     de autómata con un único estado inicial y un único estado de aceptación:
        - hoja 'a'  : inicio --a--> fin
        - hoja 'ε'  : inicio --ε--> fin
        - r · s     : se une el fin de r con el inicio de s mediante ε
        - r | s     : nuevo inicio con ε hacia r y s; los fines van con ε a un nuevo fin
        - r *       : nuevo inicio/fin con ε; se agrega ε del fin de r a su inicio
                      y ε del nuevo inicio al nuevo fin
  2. Dibujo del grafo del AFN en pantalla (imagen PNG), mostrando el estado
     inicial, los estados adicionales, el estado de aceptación y las
     transiciones etiquetadas con su símbolo.
  3. Simulación del AFN sobre una cadena w usando cerradura-epsilon, e
     imprime "sí" si w pertenece a L(r) y "no" en caso contrario.

Uso:
    python afn_thompson.py archivo_expresiones.txt "cadena_a_probar"
    python afn_thompson.py archivo_expresiones.txt        (pide la cadena por teclado)

Requiere:
    pip install matplotlib
"""

import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dibujo import dibujar_automata

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol, imprimir_arbol, EPSILON


# Objetos del autómata
class Estado:
    """Un estado del AFN. Guarda sus transiciones salientes."""

    _contador = 0

    def __init__(self):
        self.id = Estado._contador
        Estado._contador += 1
        # lista de tuplas (simbolo, estado_destino); simbolo None = transición ε
        self.transiciones = []

    def agregar(self, simbolo, destino):
        self.transiciones.append((simbolo, destino))

    def __repr__(self):
        return f"q{self.id}"


class Fragmento:
    """Fragmento de AFN de Thompson: un estado inicial y uno de aceptación."""

    def __init__(self, inicio, aceptacion):
        self.inicio = inicio
        self.aceptacion = aceptacion


class AFN:
    """AFN completo, con su estado inicial, de aceptación y todos sus estados."""

    def __init__(self, inicio, aceptacion):
        self.inicio = inicio
        self.aceptacion = aceptacion
        self.estados = self._recolectar_estados()
        self.alfabeto = self._recolectar_alfabeto()

    def _recolectar_estados(self):
        vistos = []
        pendientes = [self.inicio]
        while pendientes:
            estado = pendientes.pop()
            if estado in vistos:
                continue
            vistos.append(estado)
            for _, destino in estado.transiciones:
                if destino not in vistos:
                    pendientes.append(destino)
        return sorted(vistos, key=lambda e: e.id)

    def _recolectar_alfabeto(self):
        simbolos = set()
        for estado in self.estados:
            for simbolo, _ in estado.transiciones:
                if simbolo is not None:
                    simbolos.add(simbolo)
        return sorted(simbolos)

    #Simulación: 
    def cerradura_epsilon(self, conjunto_estados):
        """Todos los estados alcanzables usando solo transiciones ε."""
        pila = list(conjunto_estados)
        cerradura = set(conjunto_estados)
        while pila:
            estado = pila.pop()
            for simbolo, destino in estado.transiciones:
                if simbolo is None and destino not in cerradura:
                    cerradura.add(destino)
                    pila.append(destino)
        return cerradura

    def mover(self, conjunto_estados, simbolo):
        """Estados alcanzables desde el conjunto consumiendo 'simbolo'."""
        alcanzados = set()
        for estado in conjunto_estados:
            for etiqueta, destino in estado.transiciones:
                if etiqueta == simbolo:
                    alcanzados.add(destino)
        return alcanzados

    def acepta(self, cadena, pasos=None):
        """Simula el AFN sobre 'cadena'. Retorna True si es aceptada."""
        actuales = self.cerradura_epsilon({self.inicio})
        if pasos is not None:
            pasos.append(f"  Inicio            -> estados: {_fmt(actuales)}")

        for i, caracter in enumerate(cadena):
            movidos = self.mover(actuales, caracter)
            actuales = self.cerradura_epsilon(movidos)
            if pasos is not None:
                pasos.append(f"  Leyendo '{caracter}' (pos {i}) -> estados: {_fmt(actuales)}")
            if not actuales:
                if pasos is not None:
                    pasos.append("  No hay estados alcanzables: la cadena se rechaza aquí.")
                return False

        aceptada = self.aceptacion in actuales
        if pasos is not None:
            pasos.append(
                f"  Fin de la cadena. ¿Contiene el estado de aceptación "
                f"{self.aceptacion}? -> {'sí' if aceptada else 'no'}"
            )
        return aceptada


def _fmt(conjunto):
    return "{" + ", ".join(str(e) for e in sorted(conjunto, key=lambda x: x.id)) + "}"


# Construcción de Thompson desde el árbol sintáctico
def thompson(nodo, pasos):
    """
    Recorre el árbol sintáctico en postorden y arma el AFN de Thompson.
    Retorna un Fragmento (inicio, aceptacion).
    """
    #  Caso base: hoja (símbolo o ε)
    if nodo.es_hoja():
        inicio = Estado()
        fin = Estado()
        if nodo.valor == EPSILON:
            inicio.agregar(None, fin)
            pasos.append(f"  hoja 'ε' -> {inicio} --ε--> {fin}")
        else:
            inicio.agregar(nodo.valor, fin)
            pasos.append(f"  hoja '{nodo.valor}' -> {inicio} --{nodo.valor}--> {fin}")
        return Fragmento(inicio, fin)

    # Concatenación: r · s
    if nodo.valor == "·":
        frag_izq = thompson(nodo.izquierdo, pasos)
        frag_der = thompson(nodo.derecho, pasos)
        frag_izq.aceptacion.agregar(None, frag_der.inicio)
        pasos.append(
            f"  concatenación '·' -> se une {frag_izq.aceptacion} --ε--> {frag_der.inicio}"
        )
        return Fragmento(frag_izq.inicio, frag_der.aceptacion)

    # Alternancia: r | s
    if nodo.valor == "|":
        frag_izq = thompson(nodo.izquierdo, pasos)
        frag_der = thompson(nodo.derecho, pasos)
        inicio = Estado()
        fin = Estado()
        inicio.agregar(None, frag_izq.inicio)
        inicio.agregar(None, frag_der.inicio)
        frag_izq.aceptacion.agregar(None, fin)
        frag_der.aceptacion.agregar(None, fin)
        pasos.append(
            f"  alternancia '|' -> nuevo inicio {inicio} con ε a ambas ramas, "
            f"ambas ramas con ε a {fin}"
        )
        return Fragmento(inicio, fin)

    # Clausura de Kleene: r*
    if nodo.valor == "*":
        frag = thompson(nodo.izquierdo, pasos)
        inicio = Estado()
        fin = Estado()
        inicio.agregar(None, frag.inicio)   # entrar al fragmento
        inicio.agregar(None, fin)           # saltarlo (cero repeticiones)
        frag.aceptacion.agregar(None, frag.inicio)  # repetir
        frag.aceptacion.agregar(None, fin)          # salir
        pasos.append(
            f"  clausura '*' -> nuevo inicio {inicio} y fin {fin}, "
            f"con ε de salto y ε de repetición"
        )
        return Fragmento(inicio, fin)

    raise ErrorExpresion(f"Operador '{nodo.valor}' no soportado en la construcción de Thompson.")


def construir_afn(raiz, pasos):
    fragmento = thompson(raiz, pasos)
    return AFN(fragmento.inicio, fragmento.aceptacion)


# Dibujo del grafo del AFN
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


def dibujar_afn(afn, titulo, ruta_imagen):
    """
    Dibuja el AFN por niveles (distancia en transiciones desde el inicio).
    Marca el estado inicial con una flecha entrante y el de aceptación con
    doble círculo. Las transiciones ε se etiquetan con el símbolo ε.
    """
    nivel = {afn.inicio: 0}
    cola = [afn.inicio]
    while cola:
        estado = cola.pop(0)
        for _, destino in estado.transiciones:
            if destino not in nivel:
                nivel[destino] = nivel[estado] + 1
                cola.append(destino)
    for estado in afn.estados:
        nivel.setdefault(estado, 0)

    por_nivel = {}
    for estado in afn.estados:
        por_nivel.setdefault(nivel[estado], []).append(estado)

    posiciones = {}
    for nivel_actual, estados in por_nivel.items():
        estados.sort(key=lambda e: e.id)
        for indice, estado in enumerate(estados):
            desplazamiento = (len(estados) - 1) / 2
            posiciones[estado] = (nivel_actual * 2.3, (indice - desplazamiento) * 1.7)

    transiciones = [
        (estado, "\u03b5" if simbolo is None else simbolo, destino)
        for estado in afn.estados
        for simbolo, destino in estado.transiciones
    ]
    nombres = {estado: str(estado) for estado in afn.estados}

    dibujar_automata(titulo, ruta_imagen, posiciones, transiciones,
                     afn.inicio, {afn.aceptacion}, nombres)


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
        pasos_arbol = []
        try:
            raiz = construir_arbol(postfix_tokens, pasos_arbol)
        except ErrorExpresion as error:
            print(f"  ERROR construyendo el árbol: {error}")
            print("=" * 72)
            continue
        print("\nÁrbol sintáctico:")
        imprimir_arbol(raiz)

        # 3) árbol -> AFN (Thompson)
        print("\nConstrucción de Thompson (pasos):")
        pasos_thompson = []
        try:
            afn = construir_afn(raiz, pasos_thompson)
        except ErrorExpresion as error:
            print(f"  ERROR construyendo el AFN: {error}")
            print("=" * 72)
            continue
        for paso in pasos_thompson:
            print(paso)

        print(f"\nAFN generado:")
        print(f"  Estado inicial     : {afn.inicio}")
        print(f"  Estado de aceptación: {afn.aceptacion}")
        print(f"  Total de estados   : {len(afn.estados)}")
        print(f"  Alfabeto           : {{{', '.join(afn.alfabeto)}}}")
        print("  Transiciones:")
        for estado in afn.estados:
            for simbolo, destino in estado.transiciones:
                etiqueta = "ε" if simbolo is None else simbolo
                print(f"    {estado} --{etiqueta}--> {destino}")

        # 4) dibujo
        ruta_imagen = f"afn_{numero}.png"
        dibujar_afn(afn, f"AFN de Thompson para: {expresion}", ruta_imagen)
        print(f"\n  Grafo del AFN guardado en: {ruta_imagen}")

        # 5) simulación
        print(f"\nSimulación del AFN con w = '{cadena_w}':")
        pasos_sim = []
        aceptada = afn.acepta(cadena_w, pasos_sim)
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
        print('Uso: python afn_thompson.py <archivo.txt> ["cadena_w"]')
        sys.exit(1)

    procesar_archivo(ruta_archivo, cadena_w)


if __name__ == "__main__":
    main()
