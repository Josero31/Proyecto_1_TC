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
     transiciones etiquetadas con su símbolo. Los estados se numeran y se
     ubican en el dibujo siguiendo el mismo orden en que Thompson los
     construyó (izquierda a derecha para "·", ramas de "|" apiladas, "*"
     anidado), y cada expresión del archivo reinicia su numeración en q0.
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
    Retorna (Fragmento, estructura), donde 'estructura' es un árbol paralelo
    que conserva el orden real de construcción (para poder dibujar el AFN
    en ese mismo orden, en vez de reordenarlo por id o por nivel BFS).
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
        return Fragmento(inicio, fin), ("hoja", inicio, fin)

    # ---- Concatenación: r · s
    if nodo.valor == "·":
        frag_izq, est_izq = thompson(nodo.izquierdo, pasos)
        frag_der, est_der = thompson(nodo.derecho, pasos)
        frag_izq.aceptacion.agregar(None, frag_der.inicio)
        pasos.append(
            f"  concatenación '·' -> se une {frag_izq.aceptacion} --ε--> {frag_der.inicio}"
        )
        return Fragmento(frag_izq.inicio, frag_der.aceptacion), ("concat", est_izq, est_der)

    # ---- Alternancia: r | s
    if nodo.valor == "|":
        # 'inicio' se crea ANTES de recorrer las ramas para que su id (y por
        # lo tanto su posición al dibujar) quede a la izquierda de ellas,
        # igual que en el AFN resultante.
        inicio = Estado()
        frag_izq, est_izq = thompson(nodo.izquierdo, pasos)
        frag_der, est_der = thompson(nodo.derecho, pasos)
        fin = Estado()
        inicio.agregar(None, frag_izq.inicio)
        inicio.agregar(None, frag_der.inicio)
        frag_izq.aceptacion.agregar(None, fin)
        frag_der.aceptacion.agregar(None, fin)
        pasos.append(
            f"  alternancia '|' -> nuevo inicio {inicio} con ε a ambas ramas, "
            f"ambas ramas con ε a {fin}"
        )
        return Fragmento(inicio, fin), ("alt", inicio, fin, est_izq, est_der)

    # ---- Clausura de Kleene: r*
    if nodo.valor == "*":
        inicio = Estado()  # creado antes de recorrer, ver comentario arriba
        frag, est_interna = thompson(nodo.izquierdo, pasos)
        fin = Estado()
        inicio.agregar(None, frag.inicio)   # entrar al fragmento
        inicio.agregar(None, fin)           # saltarlo (cero repeticiones)
        frag.aceptacion.agregar(None, frag.inicio)  # repetir
        frag.aceptacion.agregar(None, fin)          # salir
        pasos.append(
            f"  clausura '*' -> nuevo inicio {inicio} y fin {fin}, "
            f"con ε de salto y ε de repetición"
        )
        return Fragmento(inicio, fin), ("star", inicio, fin, est_interna)

    raise ErrorExpresion(f"Operador '{nodo.valor}' no soportado en la construcción de Thompson.")


def construir_afn(raiz, pasos):
    fragmento, estructura = thompson(raiz, pasos)
    return AFN(fragmento.inicio, fragmento.aceptacion), estructura


# ---------------------------------------------------------------------------
# Cálculo de posiciones siguiendo el orden real de construcción
# ---------------------------------------------------------------------------
def _peso(estructura):
    """'Peso' vertical de un fragmento: cuántas ramas paralelas (de |) contiene."""
    tipo = estructura[0]
    if tipo == "hoja":
        return 1
    if tipo == "concat":
        return max(_peso(estructura[1]), _peso(estructura[2]))
    if tipo == "alt":
        return _peso(estructura[3]) + _peso(estructura[4])
    if tipo == "star":
        return _peso(estructura[3])
    raise ValueError(tipo)


def _layout(estructura, x, y_centro, alto_banda, posiciones):
    """
    Asigna coordenadas (x, y) a cada estado siguiendo el mismo orden en que
    Thompson construyó el fragmento: izquierda -> derecha para concatenación,
    ramas apiladas verticalmente para alternancia, y anidado para '*'.
    Retorna la coordenada x donde termina este fragmento.
    """
    tipo = estructura[0]

    if tipo == "hoja":
        _, inicio, fin = estructura
        posiciones[inicio] = (x, y_centro)
        posiciones[fin] = (x + 1.0, y_centro)
        return x + 2.0

    if tipo == "concat":
        _, est_izq, est_der = estructura
        x = _layout(est_izq, x, y_centro, alto_banda, posiciones)
        x = _layout(est_der, x, y_centro, alto_banda, posiciones)
        return x

    if tipo == "alt":
        _, inicio, fin, est_izq, est_der = estructura
        peso_izq = _peso(est_izq)
        peso_der = _peso(est_der)
        total = peso_izq + peso_der
        alto_izq = alto_banda * peso_izq / total
        alto_der = alto_banda * peso_der / total
        y_izq = y_centro + (alto_banda - alto_izq) / 2
        y_der = y_centro - (alto_banda - alto_der) / 2

        posiciones[inicio] = (x, y_centro)
        # x1 / x2 ya son la próxima columna libre después de cada rama
        # (es decir, la posición del fin de esa rama + 1), así que el fin
        # de esta alternancia se ubica ahí, sin pisar a las ramas.
        x1 = _layout(est_izq, x + 1.0, y_izq, alto_izq, posiciones)
        x2 = _layout(est_der, x + 1.0, y_der, alto_der, posiciones)
        x_max = max(x1, x2)
        posiciones[fin] = (x_max, y_centro)
        return x_max + 1.0

    if tipo == "star":
        _, inicio, fin, est_interna = estructura
        posiciones[inicio] = (x, y_centro)
        x1 = _layout(est_interna, x + 1.0, y_centro, alto_banda, posiciones)
        posiciones[fin] = (x1, y_centro)
        return x1 + 1.0

    raise ValueError(tipo)


def calcular_posiciones(estructura):
    posiciones = {}
    _layout(estructura, 0.0, 0.0, max(_peso(estructura), 1), posiciones)
    return posiciones


# ---------------------------------------------------------------------------
# Dibujo del grafo del AFN
# ---------------------------------------------------------------------------
def dibujar_afn(afn, estructura, titulo, ruta_imagen):
    """
    Dibuja el AFN siguiendo el mismo orden en que Thompson lo construyó:
    izquierda -> derecha para concatenación, ramas de '|' apiladas
    verticalmente y '*' anidado alrededor de su fragmento interno. Así el
    dibujo queda ordenado igual que los pasos de construcción impresos.
    Marca el estado inicial con una flecha entrante y el de aceptación con
    doble círculo.
    """
    posiciones_base = calcular_posiciones(estructura)
    posiciones = {
        estado: (x * 2.2, y * 1.6) for estado, (x, y) in posiciones_base.items()
    }

    columnas = sorted({round(x, 3) for x, _ in posiciones.values()})
    filas = sorted({round(y, 3) for _, y in posiciones.values()})
    ancho = max(3, len(columnas) * 1.6)
    alto = max(3, len(filas) * 1.3)
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
        Estado._contador = 0  # cada expresión reinicia su numeración desde q0
        pasos_thompson = []
        try:
            afn, estructura = construir_afn(raiz, pasos_thompson)
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
        dibujar_afn(afn, estructura, f"AFN de Thompson para: {expresion}", ruta_imagen)
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
