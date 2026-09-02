"""
Dibujo de autómatas con matplotlib, compartido por los tres módulos que
generan grafos (AFN de Thompson, AFD por subconjuntos y AFD mínimo).

Se centraliza aquí para que los tres se vean igual y para no repetir el mismo
código de dibujo tres veces.

Detalles del dibujo:
  - Los estados se dibujan como círculos con un radio fijo en unidades de
    datos, y los ejes se fijan con proporción 1:1, de modo que siempre se ven
    redondos (y no como óvalos aplastados cuando el autómata tiene pocos
    estados).
  - Un bucle (transición de un estado hacia sí mismo) se dibuja como un arco
    que sale del borde del círculo y regresa al borde del mismo círculo. Si un
    estado tiene varios símbolos en bucle, cada uno se coloca en un ángulo
    distinto para que no se superpongan.
  - En las transiciones entre estados distintos, la flecha arranca y termina
    en el borde de los círculos, y la etiqueta se desplaza hacia el mismo lado
    hacia el que se curva la flecha, para que las flechas de ida y vuelta
    entre dos estados no queden encimadas.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# radio del círculo de un estado, en unidades de datos
RADIO = 0.26

# colores
COLOR_INICIO = "#ffe0a3"
COLOR_ACEPTACION = "#c8f0c0"
COLOR_NORMAL = "#dceaff"


def _limites(posiciones):
    """Área que se va a mostrar: los estados más un margen para bucles y flechas."""
    xs = [x for x, _ in posiciones.values()]
    ys = [y for _, y in posiciones.values()]
    # a la izquierda hace falta más espacio por la flecha y el texto "inicio"
    x0, x1 = min(xs) - 2.0, max(xs) + 1.1
    y0, y1 = min(ys) - 1.1, max(ys) + 1.1
    return x0, x1, y0, y1


def _borde(x1, y1, x2, y2, radio_origen, radio_destino):
    """
    Punto de arranque y de llegada de una flecha entre dos estados, sobre el
    borde de cada círculo (no en el centro), para que la punta de la flecha
    quede pegada al estado destino. Los estados de aceptación tienen un radio
    mayor porque llevan doble círculo.
    """
    dx, dy = x2 - x1, y2 - y1
    distancia = math.hypot(dx, dy)
    if distancia == 0:
        return (x1, y1), (x2, y2)
    ux, uy = dx / distancia, dy / distancia
    return ((x1 + ux * radio_origen, y1 + uy * radio_origen),
            (x2 - ux * radio_destino, y2 - uy * radio_destino))


def _dibujar_bucle(ejes, x, y, angulo_grados, etiqueta, radio):
    """
    Dibuja una transición de un estado hacia sí mismo: un arco que sale del
    borde del círculo y vuelve a entrar por el borde, con su etiqueta afuera.
    'angulo_grados' indica hacia qué lado del estado se coloca el bucle.
    """
    theta = math.radians(angulo_grados)
    apertura = math.radians(52)  # separación entre el punto de salida y el de entrada

    salida = (x + radio * math.cos(theta + apertura),
              y + radio * math.sin(theta + apertura))
    entrada = (x + radio * math.cos(theta - apertura),
               y + radio * math.sin(theta - apertura))

    ejes.annotate("", xy=entrada, xytext=salida,
                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.0,
                                  connectionstyle="arc3,rad=-1.1"))

    distancia_texto = radio * 2.0
    ejes.text(x + distancia_texto * math.cos(theta),
              y + distancia_texto * math.sin(theta),
              etiqueta, fontsize=8, ha="center", va="center")


def dibujar_automata(titulo, ruta_imagen, posiciones, transiciones,
                     inicio, aceptacion, nombres):
    """
    Dibuja y guarda el grafo de un autómata.

    posiciones   dict estado -> (x, y)
    transiciones lista de (origen, etiqueta, destino)
    inicio       estado inicial
    aceptacion   conjunto (o lista) de estados de aceptación
    nombres      dict estado -> texto que se muestra dentro del círculo
    """
    aceptacion = set(aceptacion)

    def radio_de(estado):
        """Los estados de aceptación llevan doble círculo, así que son más grandes."""
        return RADIO * 1.22 if estado in aceptacion else RADIO

    x0, x1, y0, y1 = _limites(posiciones)

    # el tamaño de la figura sigue al tamaño del contenido (1 unidad ≈ 1 pulgada)
    ancho = min(26.0, max(4.0, x1 - x0))
    alto = min(14.0, max(3.0, y1 - y0))

    figura, ejes = plt.subplots(figsize=(ancho, alto))
    ejes.set_title(titulo, fontsize=11)
    ejes.axis("off")
    ejes.set_aspect("equal")
    ejes.set_xlim(x0, x1)
    ejes.set_ylim(y0, y1)

    # ---- transiciones
    # los bucles de un mismo estado se agrupan para repartirlos en ángulos
    bucles = {}
    for origen, etiqueta, destino in transiciones:
        if origen is destino:
            bucles.setdefault(origen, []).append(etiqueta)

    for estado, etiquetas in bucles.items():
        x, y = posiciones[estado]
        for indice, etiqueta in enumerate(etiquetas):
            angulo = 90 - indice * (360 / len(etiquetas))
            _dibujar_bucle(ejes, x, y, angulo, etiqueta, radio_de(estado))

    for origen, etiqueta, destino in transiciones:
        if origen is destino:
            continue
        xa, ya = posiciones[origen]
        xb, yb = posiciones[destino]

        # La curvatura es siempre del mismo signo, es decir, siempre hacia la
        # izquierda del sentido de avance de la flecha. Así, cuando hay una
        # transición de ida y otra de vuelta entre los mismos dos estados,
        # cada una se curva hacia un lado distinto y no quedan encimadas
        # (con signos opuestos ambas caerían del mismo lado).
        curvatura = 0.18

        desde, hasta = _borde(xa, ya, xb, yb,
                              radio_de(origen), radio_de(destino))
        ejes.annotate("", xy=hasta, xytext=desde,
                      arrowprops=dict(arrowstyle="->", color="gray", lw=1.0,
                                      connectionstyle=f"arc3,rad={curvatura}"))

        # la etiqueta se corre perpendicular a la flecha, del lado de la curva
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        dx, dy = xb - xa, yb - ya
        distancia = math.hypot(dx, dy) or 1.0
        # vector perpendicular unitario
        px, py = -dy / distancia, dx / distancia
        separacion = curvatura * distancia * 0.5 + 0.22
        ejes.text(mx + px * separacion, my + py * separacion, etiqueta,
                  fontsize=8, ha="center", va="center", color="#b03030")

    # ---- estados
    for estado, (x, y) in posiciones.items():
        if estado is inicio:
            color = COLOR_INICIO
        elif estado in aceptacion:
            color = COLOR_ACEPTACION
        else:
            color = COLOR_NORMAL

        ejes.add_patch(Circle((x, y), RADIO, facecolor=color,
                              edgecolor="black", zorder=3))
        if estado in aceptacion:      # doble círculo
            ejes.add_patch(Circle((x, y), RADIO * 1.22, facecolor="none",
                                  edgecolor="black", zorder=3))
        ejes.text(x, y, nombres[estado], ha="center", va="center",
                  fontsize=8, zorder=4)

    # ---- flecha del estado inicial
    xi, yi = posiciones[inicio]
    ejes.annotate("", xy=(xi - radio_de(inicio), yi), xytext=(xi - 1.15, yi),
                  arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ejes.text(xi - 1.25, yi, "inicio", fontsize=8, ha="right", va="center")

    figura.savefig(ruta_imagen, dpi=150, bbox_inches="tight")
    plt.close(figura)
