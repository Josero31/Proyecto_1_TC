# Proyecto 1 - Teoría de la Computación
# Autores: Ángel Mérida - 23661 José Sanchéz - 231221

Implementación de los algoritmos básicos para construir autómatas finitos a
partir de expresiones regulares y verificar la aceptación de cadenas en
lenguajes regulares.

## Pipeline

Dada una expresión regular r y una cadena w, el programa hace:

1. Shunting Yard: convierte r de notación infix a postfix.
2. Árbol sintáctico: construye el árbol a partir de la postfix (simplifica `+` y `?`).
3. Thompson: construye el AFN a partir del árbol.
4. Subconjuntos: convierte el AFN en un AFD determinista.
5. Minimización: reduce el AFD por refinamiento de particiones.
6. Simulación: evalúa w en el AFN, el AFD y el AFD mínimo, e indica si w ∈ L(r)
   con "sí" o "no".

El símbolo usado para la cadena vacía es `ε`.

## Archivos

- `shunting_yard.py`: tokenización, concatenación implícita e infix a postfix.
- `arbol_sintactico.py`: postfix a árbol sintáctico y su dibujo.
- `afn_thompson.py`: árbol a AFN (Thompson), dibujo y simulación.
- `afd_subconjuntos.py`: AFN a AFD (subconjuntos), dibujo y simulación.
- `afd_minimizacion.py`: AFD a AFD mínimo (refinamiento de particiones), dibujo
  y simulación.
- `myhill_nerode.py`: aplica el teorema de Myhill-Nerode con el algoritmo de
  llenado de tabla; calcula las clases de equivalencia, el índice, verifica que
  L(r) es regular y compara el resultado con la minimización por particiones.
- `dibujo.py`: dibujo de los grafos de autómatas con matplotlib, compartido
  por el AFN, el AFD y el AFD mínimo para que los tres se vean igual.
- `main.py`: orquesta el pipeline completo y genera las imágenes y la
  simulación de los tres autómatas por cada expresión.
- `verificacion_particiones_clase.py`: reconstruye el AFD resuelto a mano en
  clase (ejercicio "Algoritmo de particiones") y corre minimizar() sobre él
  para comprobar que el resultado coincide con lo obtenido en el pizarrón.
- `expresiones_arbol.txt`: una expresión regular por línea.

## Uso

    pip install matplotlib
    python main.py expresiones_arbol.txt "cadena_a_probar"

Si se omite la cadena, el programa la pide por teclado:

    python main.py expresiones_arbol.txt

Cada módulo también puede ejecutarse por separado con la misma sintaxis, por
ejemplo `python afd_minimizacion.py expresiones_arbol.txt "abb"`.

## Salida

Por cada expresión r se generan las imágenes del AFN, el AFD, el AFD mínimo y el
AFD por Myhill-Nerode, organizadas en subcarpetas por algoritmo:

    imagenes/afn/afn_N.png
    imagenes/afd/afd_N.png
    imagenes/afd_min/afd_min_N.png
    imagenes/myhill/myhill_N.png

donde N es el número de expresión. Cada imagen muestra el estado inicial, los
estados de aceptación y las transiciones etiquetadas. En consola se imprime la
simulación paso a paso de w en cada autómata y el veredicto w ∈ L(r).

## Verificación

    python verificacion_particiones_clase.py

Corre el algoritmo de minimización sobre el ejemplo resuelto en el pizarrón en
clase y compara el resultado (número de estados y qué estados se fusionan)
contra lo obtenido a mano. Sirve como evidencia de que la implementación es
correcta.
