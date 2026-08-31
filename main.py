"""
Programa principal del Proyecto 1 de Teoría de la Computación.

Ejecuta el pipeline completo sobre cada expresión regular de un archivo y
una cadena w:

    infix  --Shunting Yard-->  postfix
    postfix --árbol sintáctico--> árbol
    árbol  --Thompson-->  AFN
    AFN    --subconjuntos-->  AFD
    AFD    --refinamiento de particiones-->  AFD mínimo

Por cada expresión r genera:
    - la imagen del AFN, del AFD, del AFD mínimo y del AFD por Myhill-Nerode,
      organizadas en imagenes/afn, imagenes/afd, imagenes/afd_min e
      imagenes/myhill
    - la simulación de la cadena w en cada autómata
    - la respuesta "sí" / "no" para w ∈ L(r)

Reutiliza por completo los módulos de los laboratorios anteriores; aquí no se
reimplementa ningún algoritmo, solo se orquesta el flujo y se imprime el
reporte por cada línea del archivo.

Uso:
    python main.py archivo_expresiones.txt "cadena_a_probar"
    python main.py archivo_expresiones.txt        (pide la cadena por teclado)

Requiere:
    pip install matplotlib
"""

import os
import sys

from shunting_yard import a_postfix, ErrorExpresion
from arbol_sintactico import construir_arbol, imprimir_arbol
from afn_thompson import construir_afn, dibujar_afn
from afd_subconjuntos import construir_afd, dibujar_afd
from afd_minimizacion import minimizar, dibujar_afd_min
from myhill_nerode import (
    construir_tabla, clases_de_equivalencia, afd_desde_clases,
)

# las imágenes se guardan en imagenes/<algoritmo>/ para no llenar la raíz
CARPETA_IMAGENES = "imagenes"
SUBCARPETAS = ("afn", "afd", "afd_min", "myhill")


def preparar_carpetas():
    """Crea imagenes/ y una subcarpeta por algoritmo (si no existen)."""
    for sub in SUBCARPETAS:
        os.makedirs(os.path.join(CARPETA_IMAGENES, sub), exist_ok=True)


def procesar_expresion(numero, expresion, cadena_w):
    """Corre el pipeline completo para una sola expresión r y la cadena w."""
    print(f"\n{'=' * 72}")
    print(f"Expresión r ({numero}): {expresion}")
    print(f"Cadena w: '{cadena_w}'")
    print("=" * 72)

    # ---- 1) infix -> postfix
    try:
        postfix_tokens, _ = a_postfix(expresion)
    except ErrorExpresion as error:
        print(f"  ERROR en Shunting Yard: {error}")
        return
    print(f"\n[1] Postfix: {' '.join(str(t) for t in postfix_tokens)}")

    # ---- 2) postfix -> árbol sintáctico
    try:
        raiz = construir_arbol(postfix_tokens, [])
    except ErrorExpresion as error:
        print(f"  ERROR construyendo el árbol: {error}")
        return
    print("\n[2] Árbol sintáctico:")
    imprimir_arbol(raiz)

    # ---- 3) árbol -> AFN (Thompson)
    try:
        afn = construir_afn(raiz, [])
    except ErrorExpresion as error:
        print(f"  ERROR construyendo el AFN: {error}")
        return
    img_afn = os.path.join(CARPETA_IMAGENES, "afn", f"afn_{numero}.png")
    dibujar_afn(afn, f"AFN de Thompson para: {expresion}", img_afn)
    print(f"\n[3] AFN (Thompson): {len(afn.estados)} estados, "
          f"alfabeto {{{', '.join(afn.alfabeto)}}}")
    print(f"    Imagen: {img_afn}")

    pasos_afn = []
    acepta_afn = afn.acepta(cadena_w, pasos_afn)
    print("    Simulación del AFN:")
    for paso in pasos_afn:
        print("  " + paso)

    # ---- 4) AFN -> AFD (subconjuntos)
    afd = construir_afd(afn, [])
    img_afd = os.path.join(CARPETA_IMAGENES, "afd", f"afd_{numero}.png")
    dibujar_afd(afd, f"AFD (subconjuntos) para: {expresion}", img_afd)
    print(f"\n[4] AFD (subconjuntos): {len(afd.estados)} estados")
    print(f"    Imagen: {img_afd}")

    pasos_afd = []
    acepta_afd = afd.acepta(cadena_w, pasos_afd)
    print("    Simulación del AFD:")
    for paso in pasos_afd:
        print("  " + paso)

    # ---- 5) AFD -> AFD mínimo (refinamiento de particiones)
    afd_min = minimizar(afd, [])
    img_min = os.path.join(CARPETA_IMAGENES, "afd_min", f"afd_min_{numero}.png")
    dibujar_afd_min(afd_min, f"AFD mínimo para: {expresion}", img_min)
    print(f"\n[5] AFD mínimo: {len(afd_min.estados)} estados "
          f"(reducido desde {len(afd.estados)})")
    print(f"    Imagen: {img_min}")

    pasos_min = []
    acepta_min = afd_min.acepta(cadena_w, pasos_min)
    print("    Simulación del AFD mínimo:")
    for paso in pasos_min:
        print("  " + paso)

    # ---- 6) Myhill-Nerode (llenado de tabla) sobre el mismo AFD
    estados_mn, distinguible, hay_muerto = construir_tabla(afd, [])
    clases = clases_de_equivalencia(estados_mn, distinguible)
    afd_mn = afd_desde_clases(afd, clases, hay_muerto)
    indice = len(clases)
    clases_vivas = indice - (1 if hay_muerto else 0)
    img_mn = os.path.join(CARPETA_IMAGENES, "myhill", f"myhill_{numero}.png")
    dibujar_afd_min(afd_mn, f"AFD mínimo (Myhill-Nerode) para: {expresion}", img_mn)
    acepta_mn = afd_mn.acepta(cadena_w, [])
    print(f"\n[6] Myhill-Nerode (llenado de tabla): índice = {indice} clases "
          f"({clases_vivas} vivas{' + 1 muerta' if hay_muerto else ''})")
    print(f"    Índice finito -> L(r) es regular. "
          f"Clases vivas {clases_vivas} = AFD mínimo {len(afd_min.estados)} "
          f"({'coinciden' if clases_vivas == len(afd_min.estados) else 'NO COINCIDEN'})")
    print(f"    Imagen: {img_mn}")

    # ---- veredicto: los cuatro autómatas deben coincidir
    def sn(valor):
        return "sí" if valor else "no"

    print(f"\n[7] ¿w ∈ L(r)?")
    print(f"    AFN                  -> {sn(acepta_afn)}")
    print(f"    AFD                  -> {sn(acepta_afd)}")
    print(f"    AFD mínimo           -> {sn(acepta_min)}")
    print(f"    Myhill-Nerode        -> {sn(acepta_mn)}")
    coinciden = acepta_afn == acepta_afd == acepta_min == acepta_mn
    print(f"    Resultado  -> {sn(acepta_min)}"
          f"{'' if coinciden else '   (ADVERTENCIA: los autómatas no coinciden)'}")


def procesar_archivo(ruta_archivo, cadena_w):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            lineas = [linea.rstrip("\n") for linea in archivo]
    except FileNotFoundError:
        print(f"ERROR: no se encontró el archivo '{ruta_archivo}'.")
        sys.exit(1)

    print(f"Archivo a procesar: {ruta_archivo}")
    print(f"Cadena w a evaluar: '{cadena_w}'")

    preparar_carpetas()

    numero = 0
    for expresion in lineas:
        if expresion.strip() == "":
            continue
        numero += 1
        procesar_expresion(numero, expresion, cadena_w)

    print(f"\n{'=' * 72}")
    print(f"Listo. Se procesaron {numero} expresión(es).")


def main():
    if len(sys.argv) == 3:
        ruta_archivo = sys.argv[1]
        cadena_w = sys.argv[2]
    elif len(sys.argv) == 2:
        ruta_archivo = sys.argv[1]
        cadena_w = input("Ingrese la cadena w a evaluar: ")
    else:
        print('Uso: python main.py <archivo.txt> ["cadena_w"]')
        sys.exit(1)

    procesar_archivo(ruta_archivo, cadena_w)


if __name__ == "__main__":
    main()
