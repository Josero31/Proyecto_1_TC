"""
Árbol sintáctico de expresiones regulares a partir de su forma postfix.

Este programa REUTILIZA el módulo shunting_yard.py (laboratorio anterior)
para convertir cada expresión infix a postfix, y luego:

  1. Construye el árbol sintáctico usando una pila:
       - operando  -> se crea un nodo hoja y se apila
       - operador unario  (*, +, ?) -> se desapila 1 subárbol
       - operador binario (|, ·)    -> se desapilan 2 subárboles
  2. Aplica la SIMPLIFICACIÓN de las extensiones:
       r+  ==>  r · r*      (una r obligatoria seguida de cero o más r)
       r?  ==>  r | ε       (r o la cadena vacía)
     de modo que el árbol final solo contiene los operadores básicos
     ( · , | , * ) y hojas.
  3. Imprime el árbol en consola (modo texto) y lo dibuja en pantalla
     con matplotlib, guardando además una imagen PNG por expresión.

Uso:
    python arbol_sintactico.py archivo.txt

Requiere:
    pip install matplotlib
    (y tener shunting_yard.py en la misma carpeta)
"""

import sys
import matplotlib
matplotlib.use("Agg")  # genera las imágenes sin necesitar ventana gráfica
import matplotlib.pyplot as plt

from shunting_yard import (
    a_postfix, ErrorExpresion,
    LITERAL, ALT, CONCAT, QUANT, Token,
)

EPSILON = "ε"


# Nodo del árbol sintáctico
class Nodo:
    """Objeto que guarda la información de cada nodo del árbol sintáctico."""

    _contador = 0  # para asignar un id único a cada nodo (útil al dibujar)

    def __init__(self, valor, izquierdo=None, derecho=None):
        self.valor = valor          # símbolo del nodo: 'a', '·', '|', '*', 'ε', '[ae]'...
        self.izquierdo = izquierdo  # subárbol izquierdo (None en hojas)
        self.derecho = derecho      # subárbol derecho  (None en hojas y unarios)
        Nodo._contador += 1
        self.id = Nodo._contador

    def es_hoja(self):
        return self.izquierdo is None and self.derecho is None

    def clonar(self):
        """Copia profunda del subárbol (necesaria para la expansión de r+)."""
        izq = self.izquierdo.clonar() if self.izquierdo else None
        der = self.derecho.clonar() if self.derecho else None
        return Nodo(self.valor, izq, der)



# Construcción del árbol desde la postfix (con pila)
def construir_arbol(postfix_tokens, pasos):
    """
    Recorre la lista de tokens en postfix y construye el árbol sintáctico
    usando una pila de subárboles. Aplica las simplificaciones de '+' y '?'.
    Agrega a 'pasos' una línea de traza por cada operación.
    """
    pila = []

    def estado():
        return "[" + " ".join(f"({n.valor})" for n in pila) + "]"

    for token in postfix_tokens:
        if token.tipo == LITERAL:
            pila.append(Nodo(token.valor))
            pasos.append(f"  '{token.valor}' -> hoja nueva            | pila de árboles: {estado()}")

        elif token.tipo == QUANT:
            if not pila:
                raise ErrorExpresion(f"Operador '{token.valor}' sin operando.")
            hijo = pila.pop()

            if token.valor == "*":
                nodo = Nodo("*", hijo)
                pasos.append(f"  '*' -> nodo estrella        | pila de árboles: ...")

            elif token.valor == "+":
                # SIMPLIFICACIÓN: r+  =  r · r*
                copia = hijo.clonar()
                nodo = Nodo("·", hijo, Nodo("*", copia))
                pasos.append(f"  '+' -> simplificado a r·r*  | pila de árboles: ...")

            elif token.valor == "?":
                # SIMPLIFICACIÓN: r?  =  r | ε
                nodo = Nodo("|", hijo, Nodo(EPSILON))
                pasos.append(f"  '?' -> simplificado a r|ε   | pila de árboles: ...")

            else:
                # cuantificador {n,m}: se deja como nodo unario con su texto
                nodo = Nodo(token.valor, hijo)
                pasos.append(f"  '{token.valor}' -> nodo cuantificador | pila de árboles: ...")

            pila.append(nodo)
            pasos[-1] = pasos[-1].replace("...", estado())

        else:  # ALT o CONCAT: operadores binarios
            if len(pila) < 2:
                raise ErrorExpresion(f"Operador '{token.valor}' necesita dos operandos.")
            derecho = pila.pop()
            izquierdo = pila.pop()
            pila.append(Nodo(token.valor, izquierdo, derecho))
            pasos.append(f"  '{token.valor}' -> nodo binario         | pila de árboles: {estado()}")

    if len(pila) != 1:
        raise ErrorExpresion("La expresión postfix no produce un árbol único (está mal formada).")

    return pila[0]


# Impresión del árbol en consola (modo texto)
def imprimir_arbol(nodo, prefijo="", es_ultimo=True):
    conector = "└── " if es_ultimo else "├── "
    print(prefijo + conector + nodo.valor)
    hijos = [h for h in (nodo.izquierdo, nodo.derecho) if h is not None]
    for i, hijo in enumerate(hijos):
        extension = "    " if es_ultimo else "│   "
        imprimir_arbol(hijo, prefijo + extension, i == len(hijos) - 1)


# Dibujo del árbol con matplotlib
def _asignar_posiciones(nodo, profundidad, siguiente_x, posiciones):
    """Postorden: primero se posicionan los hijos, luego el padre se centra."""
    if nodo.izquierdo:
        _asignar_posiciones(nodo.izquierdo, profundidad + 1, siguiente_x, posiciones)
    if nodo.derecho:
        _asignar_posiciones(nodo.derecho, profundidad + 1, siguiente_x, posiciones)
    if nodo.es_hoja():
        x = siguiente_x[0]
        siguiente_x[0] += 1
    else:
        hijos = [h for h in (nodo.izquierdo, nodo.derecho) if h is not None]
        x = sum(posiciones[h.id][0] for h in hijos) / len(hijos)
    posiciones[nodo.id] = (x, -profundidad)


def dibujar_arbol(raiz, titulo, ruta_imagen):
    posiciones = {}
    _asignar_posiciones(raiz, 0, [0], posiciones)

    figura, ejes = plt.subplots(figsize=(11, 7))
    ejes.set_title(titulo, fontsize=11)
    ejes.axis("off")

    def dibujar_nodo(nodo):
        x, y = posiciones[nodo.id]
        for hijo in (nodo.izquierdo, nodo.derecho):
            if hijo is not None:
                hx, hy = posiciones[hijo.id]
                ejes.plot([x, hx], [y, hy], "-", color="gray", zorder=1)
                dibujar_nodo(hijo)
        color = "#cfe8ff" if not nodo.es_hoja() else "#d8f5d0"
        ejes.scatter([x], [y], s=900, c=color, edgecolors="black", zorder=2)
        ejes.text(x, y, nodo.valor, ha="center", va="center", fontsize=10, zorder=3)

    dibujar_nodo(raiz)
    figura.tight_layout()
    figura.savefig(ruta_imagen, dpi=150)
    plt.close(figura)


# Procesamiento del archivo
def procesar_archivo(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            lineas = [linea.rstrip("\n") for linea in archivo]
    except FileNotFoundError:
        print(f"ERROR: no se encontró el archivo '{ruta_archivo}'.")
        sys.exit(1)

    print(f"Archivo a procesar: {ruta_archivo}")
    print(f"Total de líneas encontradas: {len(lineas)}")
    print("=" * 70)

    numero_expresion = 0
    for numero_linea, expresion in enumerate(lineas, start=1):
        if expresion.strip() == "":
            continue
        numero_expresion += 1

        print(f"\nLínea {numero_linea} (infix): {expresion}")
        print("-" * 70)

        # --- Etapa 1: infix -> postfix (reutilizando el laboratorio anterior)
        try:
            postfix_tokens, pasos_sy = a_postfix(expresion)
        except ErrorExpresion as error:
            print(f"  ERROR en Shunting Yard: {error}")
            print("=" * 70)
            continue

        print("Pasos del algoritmo de Shunting Yard (infix -> postfix):")
        for paso in pasos_sy:
            print(paso)
        postfix_str = " ".join(str(t) for t in postfix_tokens)
        print(f"\nExpresión en postfix: {postfix_str}")

        # --- Etapa 2: postfix -> árbol sintáctico
        print("\nConstrucción del árbol sintáctico (postfix -> árbol):")
        pasos_arbol = []
        try:
            raiz = construir_arbol(postfix_tokens, pasos_arbol)
        except ErrorExpresion as error:
            print(f"  ERROR construyendo el árbol: {error}")
            print("=" * 70)
            continue

        for paso in pasos_arbol:
            print(paso)

        print("\nÁrbol sintáctico (vista en consola):")
        imprimir_arbol(raiz)

        # --- Etapa 3: dibujo del árbol
        ruta_imagen = f"arbol_{numero_expresion}.png"
        dibujar_arbol(raiz, f"Árbol sintáctico de: {expresion}", ruta_imagen)
        print(f"\nÁrbol dibujado y guardado en: {ruta_imagen}")
        print("=" * 70)


def main():
    if len(sys.argv) != 2:
        print("Uso: python arbol_sintactico.py <ruta_del_archivo.txt>")
        sys.exit(1)
    procesar_archivo(sys.argv[1])


if __name__ == "__main__":
    main()
