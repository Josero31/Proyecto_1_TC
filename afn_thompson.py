"""
AFN por construcción de Thompson a partir del árbol sintáctico de una regex.

Este programa REUTILIZA los laboratorios anteriores:
    shunting_yard.py    -> infix a postfix
    arbol_sintactico.py -> postfix a árbol sintáctico (con simplificación de + y ?)

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol, imprimir_arbol, EPSILON


# ---------------------------------------------------------------------------
# Objetos del autómata
# ---------------------------------------------------------------------------
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

    # ---------------- simulación ----------------
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


# ---------------------------------------------------------------------------
# Construcción de Thompson desde el árbol sintáctico
# ---------------------------------------------------------------------------
def thompson(nodo, pasos):
    """
    Recorre el árbol sintáctico en postorden y arma el AFN de Thompson.
    Retorna un Fragmento (inicio, aceptacion).
    """
    # ---- Caso base: hoja (símbolo o ε)
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

    # ---- Concatenación: r · s
    if nodo.valor == "·":
        frag_izq = thompson(nodo.izquierdo, pasos)
        frag_der = thompson(nodo.derecho, pasos)
        frag_izq.aceptacion.agregar(None, frag_der.inicio)
        pasos.append(
            f"  concatenación '·' -> se une {frag_izq.aceptacion} --ε--> {frag_der.inicio}"
        )
        return Fragmento(frag_izq.inicio, frag_der.aceptacion)

    # ---- Alternancia: r | s
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

    # ---- Clausura de Kleene: r*
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


# ---------------------------------------------------------------------------
# Dibujo del grafo del AFN
# ---------------------------------------------------------------------------
def dibujar_afn(afn, titulo, ruta_imagen):
    """
    Dibuja el AFN por niveles (distancia en transiciones desde el inicio).
    Marca el estado inicial con una flecha entrante y el de aceptación con
    doble círculo.
    """
    # nivel de cada estado = distancia mínima desde el inicio (BFS)
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

    # agrupar por nivel y asignar coordenadas
    por_nivel = {}
    for estado in afn.estados:
        por_nivel.setdefault(nivel[estado], []).append(estado)

    posiciones = {}
    for nivel_actual, estados in por_nivel.items():
        estados.sort(key=lambda e: e.id)
        for indice, estado in enumerate(estados):
            desplazamiento = (len(estados) - 1) / 2
            posiciones[estado] = (nivel_actual * 2.2, (indice - desplazamiento) * 1.6)

    ancho = max(3, len(por_nivel) * 1.6)
    alto = max(3, max(len(v) for v in por_nivel.values()) * 1.3)
    figura, ejes = plt.subplots(figsize=(min(ancho, 26), min(alto, 14)))
    ejes.set_title(titulo, fontsize=11)
    ejes.axis("off")

    # transiciones
    for estado in afn.estados:
        x1, y1 = posiciones[estado]
        for simbolo, destino in estado.transiciones:
            x2, y2 = posiciones[destino]
            etiqueta = "ε" if simbolo is None else simbolo
            if estado is destino:
                ejes.annotate("", xy=(x1, y1 + 0.35), xytext=(x1 + 0.5, y1 + 0.9),
                              arrowprops=dict(arrowstyle="->", color="gray",
                                              connectionstyle="arc3,rad=0.6"))
                ejes.text(x1 + 0.5, y1 + 1.0, etiqueta, fontsize=8, ha="center")
                continue
            curvatura = 0.18 if y1 <= y2 else -0.18
            ejes.annotate("", xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(arrowstyle="->", color="gray", lw=1.0,
                                          shrinkA=13, shrinkB=13,
                                          connectionstyle=f"arc3,rad={curvatura}"))
            ejes.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, etiqueta,
                      fontsize=8, ha="center", color="#b03030")

    # estados
    for estado in afn.estados:
        x, y = posiciones[estado]
        if estado is afn.inicio:
            color = "#ffe0a3"
        elif estado is afn.aceptacion:
            color = "#c8f0c0"
        else:
            color = "#dceaff"
        ejes.scatter([x], [y], s=620, c=color, edgecolors="black", zorder=3)
        if estado is afn.aceptacion:  # doble círculo
            ejes.scatter([x], [y], s=980, facecolors="none", edgecolors="black", zorder=3)
        ejes.text(x, y, str(estado), ha="center", va="center", fontsize=8, zorder=4)

    # flecha del estado inicial
    xi, yi = posiciones[afn.inicio]
    ejes.annotate("", xy=(xi - 0.28, yi), xytext=(xi - 1.1, yi),
                  arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ejes.text(xi - 1.2, yi, "inicio", fontsize=8, ha="right", va="center")

    ejes.relim()
    ejes.autoscale_view()
    x0, x1 = ejes.get_xlim()
    y0, y1 = ejes.get_ylim()
    ejes.set_xlim(x0 - 0.4, x1 + 0.4)
    ejes.set_ylim(y0 - 0.4, y1 + 0.8)

    figura.tight_layout()
    figura.savefig(ruta_imagen, dpi=150, bbox_inches="tight")
    plt.close(figura)


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
