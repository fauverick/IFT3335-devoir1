"""
Rendu et animation de la recherche sur la carte.

BOÎTE NOIRE — Ce fichier vous est fourni ; l'affichage n'est pas évalué. Il
sert à voir ce que font vos algorithmes : les nœuds s'allument dans l'ordre
où ils sont développés, puis le chemin retenu est tracé.

C'est là que la comparaison des quatre algorithmes devient évidente, la forme
de la tache d'exploration trahissant la stratégie employée :

    BFS   tache ronde et dense, qui grandit couche par couche.
    DFS   filament unique qui part au hasard et serpente très loin.
    UCS   tache ronde également : Dijkstra ignore où est l'arrivée.
    A*    tache ALLONGÉE, tendue vers la destination — c'est l'heuristique
          qui l'étire, et c'est tout le gain de l'algorithme.

Une tache circulaire pour A* signifie que votre heuristique n'oriente rien :
votre A* s'est réduit à UCS.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button

from road_graph import RoadGraph, SearchResult

# Palette : fond sobre, réseau discret, exploration chaude, chemin saillant.
COULEUR_FOND = "#111418"
COULEUR_RESEAU = "#6b7c8f"
EPAISSEUR_RESEAU = 1.1
COULEUR_BOUTON = "#2f3944"
COULEUR_EXPLORE = "#f2a25c"
COULEUR_CHEMIN = "#4fd1c5"
COULEUR_DEPART = "#7bd88f"
COULEUR_ARRIVEE = "#ff6b6b"
COULEUR_TEXTE = "#e6edf3"
#: Nœuds où l'heuristique a annoncé PLUS que le coût réel restant.
COULEUR_SURESTIME = "#c74dff"

#: Nombre maximal d'images pour la phase d'exploration.
IMAGES_EXPLORATION = 120
#: Nombre d'images pour le tracé du chemin final.
IMAGES_CHEMIN = 25
#: Images par seconde de l'animation.
FPS = 20


def _geometrie_arete(rg: RoadGraph, u: int, v: int) -> list[tuple[float, float]]:
    """Points (lon, lat) dessinant l'arête u -> v, sa courbure comprise."""
    geometrie = rg.edge_data(u, v).get("geometry")
    if geometrie is not None:
        return list(geometrie.coords)
    (lat_u, lon_u), (lat_v, lon_v) = rg.coords[u], rg.coords[v]
    return [(lon_u, lat_u), (lon_v, lat_v)]


def _segments_reseau(rg: RoadGraph) -> list[list[tuple[float, float]]]:
    return [_geometrie_arete(rg, u, v) for u, voisins in rg.adj_dist.items() for v, _ in voisins]


def _segments_chemin(rg: RoadGraph, path: list[int]) -> list[list[tuple[float, float]]]:
    return [_geometrie_arete(rg, u, v) for u, v in zip(path, path[1:])]


def _dessiner_fond(ax, rg: RoadGraph, titre: str, taille_titre: int = 13) -> None:
    """Trace le réseau routier et cadre la vue sur des axes déjà créés."""
    ax.set_facecolor(COULEUR_FOND)
    ax.add_collection(
        LineCollection(_segments_reseau(rg), colors=COULEUR_RESEAU, linewidths=EPAISSEUR_RESEAU,
                       zorder=1)
    )

    lats = [lat for lat, _ in rg.coords.values()]
    lons = [lon for _, lon in rg.coords.values()]
    marge_lat = (max(lats) - min(lats)) * 0.02
    marge_lon = (max(lons) - min(lons)) * 0.02
    ax.set_xlim(min(lons) - marge_lon, max(lons) + marge_lon)
    ax.set_ylim(min(lats) - marge_lat, max(lats) + marge_lat)
    # Un degré de longitude est plus court qu'un degré de latitude : sans
    # cette correction, la carte apparaîtrait étirée horizontalement.
    ax.set_aspect(1 / math.cos(math.radians(sum(lats) / len(lats))))
    ax.set_axis_off()
    ax.set_title(titre, color=COULEUR_TEXTE, fontsize=taille_titre, pad=14)


def _preparer_figure(rg: RoadGraph, titre: str):
    """Trace le fond de carte sur une figure neuve et renvoie (figure, axes)."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=COULEUR_FOND)
    _dessiner_fond(ax, rg, titre)
    fig.tight_layout()
    return fig, ax


def _marquer_extremites(
    ax, rg: RoadGraph, start: int, goal: int, legende: bool = True
) -> None:
    lat_d, lon_d = rg.coords[start]
    lat_a, lon_a = rg.coords[goal]
    ax.scatter([lon_d], [lat_d], s=280, c=COULEUR_DEPART, marker="o",
               edgecolors="white", linewidths=2.0, zorder=6, label="Départ")
    ax.scatter([lon_a], [lat_a], s=480, c=COULEUR_ARRIVEE, marker="*",
               edgecolors="white", linewidths=1.6, zorder=6, label="Arrivée")
    if not legende:
        return
    legende_ax = ax.legend(loc="lower right", facecolor=COULEUR_FOND,
                           edgecolor=COULEUR_RESEAU, markerscale=0.6,
                           labelspacing=1.0, borderpad=0.8)
    for texte in legende_ax.get_texts():
        texte.set_color(COULEUR_TEXTE)


def _texte_etat(ax, taille: int = 10) -> "plt.Text":
    return ax.text(
        0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
        color=COULEUR_TEXTE, fontsize=taille, family="monospace", zorder=7,
    )


def _marquer_surestimations(
    ax, rg: RoadGraph, fautifs: set[int] | None, taille: float = 34
) -> bool:
    """
    Surligne les nœuds où l'heuristique a dépassé le coût réel restant.

    Renvoie True si quelque chose a été tracé. Rien n'apparaît quand
    l'heuristique ne surestime jamais : sur une carte propre, il n'y a
    simplement aucune croix.
    """
    if not fautifs:
        return False
    points = [rg.coords[n] for n in fautifs if n in rg.coords]
    if not points:
        return False
    ax.scatter([lon for _, lon in points], [lat for lat, _ in points],
               s=taille, c=COULEUR_SURESTIME, marker="x", linewidths=1.4,
               zorder=4, label="h > coût réel")
    return True


def _tracer_recherche(
    ax, rg: RoadGraph, resultat: SearchResult, taille_point: float, epaisseur: float,
    fautifs: set[int] | None = None,
) -> None:
    """Nuage des nœuds développés, chemin retenu, puis nœuds surestimés."""
    if resultat.expanded:
        points = [rg.coords[n] for n in resultat.expanded]
        ax.scatter([lon for _, lon in points], [lat for lat, _ in points],
                   s=taille_point, c=COULEUR_EXPLORE, alpha=0.9, zorder=2)
    if resultat.found:
        ax.add_collection(
            LineCollection(_segments_chemin(rg, resultat.path), colors=COULEUR_CHEMIN,
                           linewidths=epaisseur, zorder=5)
        )
    _marquer_surestimations(ax, rg, fautifs)


def afficher_resultat(
    rg: RoadGraph,
    resultat: SearchResult,
    start: int,
    goal: int,
    titre: str = "Recherche sur le réseau routier",
    sortie: Path | None = None,
    fautifs: set[int] | None = None,
) -> None:
    """
    Image fixe : nœuds explorés et chemin final, sans animation.

    ``fautifs`` : nœuds où l'heuristique a annoncé plus que le coût réel
    restant ; ils sont marqués d'une croix. Rien n'apparaît si l'ensemble est
    vide, ce qui est le cas d'une heuristique admissible.
    """
    fig, ax = _preparer_figure(rg, titre)
    _tracer_recherche(ax, rg, resultat, taille_point=16, epaisseur=3.0, fautifs=fautifs)
    _marquer_extremites(ax, rg, start, goal)
    _texte_etat(ax).set_text(resultat.describe())
    _terminer(fig, sortie)


def comparer_algorithmes(
    rg: RoadGraph,
    resultats: dict[str, SearchResult],
    start: int,
    goal: int,
    titre: str = "Comparaison des algorithmes",
    sortie: Path | None = None,
) -> None:
    """
    Une carte par algorithme, sur une même grille et à la même échelle.

    ``resultats`` associe le nom à afficher au ``SearchResult`` correspondant ;
    l'ordre du dictionnaire est celui des vignettes. Mettre les recherches
    côte à côte est le seul moyen de voir d'un coup d'œil ce que les chiffres
    disent : la tache de A* est visiblement plus petite et tendue vers
    l'arrivée, celle de DFS n'est qu'un long fil égaré.
    """
    if not resultats:
        return

    nombre = len(resultats)
    colonnes = 2 if nombre > 1 else 1
    rangees = math.ceil(nombre / colonnes)
    fig, axes = plt.subplots(
        rangees, colonnes,
        figsize=(6.4 * colonnes, 6.6 * rangees),
        facecolor=COULEUR_FOND,
        squeeze=False,
    )
    cases = [ax for rangee in axes for ax in rangee]

    for ax, (nom, resultat) in zip(cases, resultats.items()):
        _dessiner_fond(ax, rg, nom, taille_titre=12)
        _tracer_recherche(ax, rg, resultat, taille_point=9, epaisseur=2.4)
        _marquer_extremites(ax, rg, start, goal, legende=False)
        _texte_etat(ax, taille=8).set_text(resultat.describe())

    for ax in cases[nombre:]:  # cases restantes d'une grille incomplète
        ax.set_axis_off()

    fig.suptitle(titre, color=COULEUR_TEXTE, fontsize=14)
    # rect : on réserve le haut de la figure au titre général, sinon
    # tight_layout le laisse chevaucher le titre des vignettes.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 / rangees))
    _terminer(fig, sortie)


def animer_recherche(
    rg: RoadGraph,
    resultat: SearchResult,
    start: int,
    goal: int,
    titre: str = "Recherche sur le réseau routier",
    sortie: Path | None = None,
    fautifs: set[int] | None = None,
) -> None:
    """
    Anime l'exploration puis le tracé du chemin.

    ``sortie`` : chemin de fichier ``.gif`` où enregistrer l'animation ; si
    ``None``, l'animation est affichée dans une fenêtre, avec un bouton
    « Rejouer » qui la relance depuis le début.
    ``fautifs`` : nœuds où l'heuristique a dépassé le coût réel restant. Ils
    apparaissent en croix AU FUR ET À MESURE que la recherche les atteint, de
    sorte qu'on voit à quel moment l'estimation dérape.
    """
    if not resultat.expanded:
        afficher_resultat(rg, resultat, start, goal, titre, sortie, fautifs)
        return

    fig, ax = _preparer_figure(rg, titre)
    _marquer_extremites(ax, rg, start, goal)
    etat = _texte_etat(ax)

    explores = ax.scatter([], [], s=16, c=COULEUR_EXPLORE, alpha=0.9, zorder=2)
    surestimes = ax.scatter([], [], s=34, c=COULEUR_SURESTIME, marker="x",
                            linewidths=1.4, zorder=4)
    trace = LineCollection([], colors=COULEUR_CHEMIN, linewidths=3.0, zorder=5)
    ax.add_collection(trace)
    a_signaler = set(fautifs or ())

    expansions = resultat.expanded
    lot = max(1, math.ceil(len(expansions) / IMAGES_EXPLORATION))
    n_images_exploration = math.ceil(len(expansions) / lot)
    segments = _segments_chemin(rg, resultat.path) if resultat.found else []
    n_images_chemin = IMAGES_CHEMIN if segments else 0

    def image(i: int):
        if i < n_images_exploration:
            vus = expansions[: (i + 1) * lot]
            points = [rg.coords[n] for n in vus]
            explores.set_offsets([(lon, lat) for lat, lon in points])
            croix = [rg.coords[n] for n in vus if n in a_signaler]
            if croix:
                surestimes.set_offsets([(lon, lat) for lat, lon in croix])
            etat.set_text(
                f"Exploration : {len(vus)} / {len(expansions)} nœuds développés"
                + (f"   ·   {len(croix)} nœud(s) où h > coût réel" if croix else "")
            )
        else:
            j = i - n_images_exploration + 1
            k = max(1, math.ceil(len(segments) * j / n_images_chemin))
            trace.set_segments(segments[:k])
            etat.set_text(resultat.describe())
        return explores, surestimes, trace, etat

    def vider():
        """Remet la carte à blanc : appelée au début de chaque lecture."""
        explores.set_offsets(np.empty((0, 2)))
        surestimes.set_offsets(np.empty((0, 2)))
        trace.set_segments([])
        etat.set_text("")
        return explores, surestimes, trace, etat

    def lancer() -> FuncAnimation:
        return FuncAnimation(
            fig,
            image,
            frames=n_images_exploration + n_images_chemin,
            init_func=vider,
            interval=1000 / FPS,
            blit=False,
            repeat=False,
        )

    animation = lancer()

    if sortie is not None:
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        animation.save(sortie, writer=PillowWriter(fps=FPS))
        print(f"Animation enregistrée : {sortie}")
        plt.close(fig)
        return

    # En haut à gauche, dans la bande du titre : le texte d'état occupe déjà
    # le coin supérieur gauche de la carte, juste en dessous.
    axe_bouton = fig.add_axes((0.015, 0.955, 0.11, 0.035))
    bouton = Button(axe_bouton, "↻ Rejouer", color=COULEUR_BOUTON, hovercolor="#45525f")
    bouton.label.set_color(COULEUR_TEXTE)

    def rejouer(_evenement) -> None:
        # Une FuncAnimation terminée ne peut pas être relancée (matplotlib la
        # détache de son minuteur) : on en crée une neuve, qui démarre au
        # prochain rafraîchissement de la fenêtre.
        nonlocal animation
        if animation.event_source is not None:  # clic pendant la lecture
            animation.pause()
        animation = lancer()
        fig.canvas.draw_idle()

    bouton.on_clicked(rejouer)
    plt.show()


def _terminer(fig, sortie: Path | None) -> None:
    if sortie is not None:
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sortie, dpi=140, facecolor=fig.get_facecolor())
        print(f"Image enregistrée : {sortie}")
        plt.close(fig)
    else:
        plt.show()
