"""
Chargement et pré-traitement du réseau routier OpenStreetMap.

BOÎTE NOIRE — Ce fichier vous est fourni. Vous n'avez ni à le modifier, ni
même à le comprendre en détail pour réussir le travail.

Son rôle : transformer le fichier GraphML extrait d'OpenStreetMap en
structures Python élémentaires, afin que votre algorithme de recherche n'ait
à connaître ni OSM, ni NetworkX, ni la géométrie des routes.

Le réseau est fourni sous forme de liste d'adjacence pondérée :

    adj_dist : dict[int, list[tuple[int, float]]]
        ``adj_dist[u] == [(v, longueur_uv), ...]``, longueurs en MÈTRES.
        C'est le problème du chemin le plus COURT.

Les nœuds sont NUMÉROTÉS de 0 à n-1 (et non par leur identifiant
OpenStreetMap, qui compte jusqu'à onze chiffres). La correspondance vers OSM
est conservée dans ``RoadGraph.osm``, à l'usage exclusif de l'affichage.

Chaque nœud est également situé dans le plan :

    coords : dict[int, tuple[float, float]]
        ``coords[n] == (latitude, longitude)``, en degrés décimaux.
        Réservé à l'affichage.

    coords_m : dict[int, tuple[float, float]]
        ``coords_m[n] == (x, y)``, en MÈTRES. C'est ce que reçoivent les
        heuristiques du solveur étudiant.

Le graphe est ORIENTÉ (les sens uniques sont respectés) : la présence de
(v, p) dans adj[u] n'implique pas la présence de (u, p) dans adj[v].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import osmnx as ox

# Types nommés, réutilisés dans les signatures du solveur étudiant.
Adjacency = dict[int, list[tuple[int, float]]]
Coords = dict[int, tuple[float, float]]
CoordsM = dict[int, tuple[float, float]]

#: Rayon moyen de la Terre, en mètres.
EARTH_RADIUS_M = 6_371_008.8

#: Facteur de sécurité appliqué aux coordonnées métriques (voir ``coords_m``).
#: Une projection plane à origine fixe surestime légèrement certaines
#: distances — jusqu'à 0,03 % sur ce réseau. Sans ce facteur, la distance à vol
#: d'oiseau entre deux nœuds voisins pourrait dépasser de quelques centimètres
#: la longueur réelle de la rue qui les relie, et l'heuristique ne serait plus
#: admissible. Le coût en force heuristique est négligeable (moins d'un mètre).
FACTEUR_ADMISSIBILITE = 0.999

#: Emplacement par défaut du sous-graphe pré-extrait.
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "roads_map.graphml"

def _distance_approx_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance APPROCHÉE entre deux points GPS, en mètres (projection plane
    locale). Suffisante pour repérer le nœud le plus proche d'un point ou
    pour afficher un ordre de grandeur, à l'échelle d'un quartier.

    Ce n'est volontairement PAS la formule exacte attendue dans votre
    heuristique : celle-ci est à écrire dans ``student_solver.py``.
    """
    lat_moyenne = math.radians((lat1 + lat2) / 2)
    dx = math.radians(lon2 - lon1) * math.cos(lat_moyenne)
    dy = math.radians(lat2 - lat1)
    return EARTH_RADIUS_M * math.hypot(dx, dy)


def format_cost(cost: float) -> str:
    """Met en forme une distance en mètres (et en kilomètres, en prime)."""
    if math.isinf(cost):
        return "infini"
    return f"{cost:.0f} m ({cost / 1000:.2f} km)"


@dataclass(frozen=True)
class SearchResult:
    """
    Résultat d'une recherche de chemin.

    Attributs
    ---------
    path :
        Liste ordonnée des identifiants de nœuds, du départ à l'arrivée (les
        deux inclus). ``None`` si aucun chemin n'existe.
    cost :
        Longueur totale du chemin, en MÈTRES.
        ``math.inf`` s'il n'y a pas de chemin.
    expanded :
        Nœuds retirés de la file de priorité et développés, dans l'ordre.
        Sert à animer la recherche et à mesurer l'efficacité de l'heuristique.
    n_explored :
        COMPTEUR de nœuds explorés, que vous tenez vous-même : un incrément
        de 1 à chaque nœud développé, c'est-à-dire au même endroit que le
        ``expanded.append(n)`` correspondant. C'est la mesure du TRAVAIL
        fourni par l'algorithme, et c'est elle qui rend visible l'apport de
        l'heuristique : sur le réseau réel, A* explore typiquement trois à
        quatre fois moins de nœuds que UCS pour trouver le même chemin
        optimal. L'écart dépend du trajet — il peut être nul sur un trajet
        très court, et dépasser un facteur 10 sur une longue traversée.

        Il doit donc toujours vérifier ``n_explored == len(expanded)`` — les
        vérifications et la correction le contrôlent. Le compteur n'est pas
        déduit de ``expanded`` : c'est à vous de l'incrémenter, y compris
        lorsqu'aucun chemin n'est trouvé.
    """

    path: list[int] | None
    cost: float
    expanded: list[int] = field(default_factory=list)
    n_explored: int = 0

    @property
    def found(self) -> bool:
        """Vrai si un chemin a été trouvé."""
        return bool(self.path)

    @property
    def compteur_coherent(self) -> bool:
        """Vrai si le compteur annoncé correspond aux nœuds réellement développés."""
        return self.n_explored == len(self.expanded)

    def describe(self) -> str:
        if not self.found:
            return f"aucun chemin trouvé ({self.n_explored} nœuds explorés)"
        return (
            f"{len(self.path)} nœuds · {format_cost(self.cost)} · "
            f"{self.n_explored} nœuds explorés"
        )

    def __str__(self) -> str:
        if not self.found:
            return f"aucun chemin trouvé ({self.n_explored} nœuds explorés)"
        return (
            f"{len(self.path)} nœuds · coût {self.cost:.1f} · "
            f"{self.n_explored} nœuds explorés"
        )


@dataclass(frozen=True)
class RoadGraph:
    """
    Réseau routier prêt à l'emploi.

    ``adj_dist`` et ``coords_m`` sont les seules données transmises au
    solveur étudiant ; ``nx_graph`` n'est conservé que pour l'affichage.
    """

    adj_dist: Adjacency
    coords: Coords
    coords_m: CoordsM
    osm: dict[int, int]
    nx_graph: nx.MultiDiGraph
    _dist: dict[tuple[int, int], float]

    # -------------------------------------------------------------- métadonnées
    @property
    def nodes(self) -> list[int]:
        return list(self.coords)

    @property
    def n_edges(self) -> int:
        return len(self._dist)

    def describe(self) -> str:
        return (
            f"{len(self.coords)} nœuds numérotés de 0 à {len(self.coords) - 1}, "
            f"{self.n_edges} arêtes orientées"
        )

    def edge_data(self, u: int, v: int) -> dict:
        """
        Attributs OpenStreetMap de l'arête u -> v : nom de rue, géométrie…

        Réservé à l'affichage : c'est le seul endroit où les numéros 0..n-1
        sont retraduits en identifiants OSM d'origine.
        """
        donnees = self.nx_graph.get_edge_data(self.osm[u], self.osm[v]) or {}
        return min(donnees.values(), key=lambda d: d.get("length", math.inf), default={})

    # ----------------------------------------------------------------- requêtes
    def nearest_node(self, lat: float, lon: float) -> int:
        """Identifiant du nœud du réseau le plus proche d'un point GPS."""
        return min(self.coords, key=lambda n: _distance_approx_m(lat, lon, *self.coords[n]))

    def edge_length(self, u: int, v: int) -> float:
        """Longueur de l'arête u -> v, en mètres."""
        return self._poids(self._dist, u, v)

    @staticmethod
    def _poids(index: dict[tuple[int, int], float], u: int, v: int) -> float:
        try:
            return index[(u, v)]
        except KeyError:
            raise ValueError(f"L'arête {u} -> {v} n'existe pas dans le réseau.") from None

    def path_length(self, path: list[int]) -> float:
        """
        Longueur réelle d'un chemin, en mètres.

        Lève ``ValueError`` si le chemin emprunte une arête inexistante :
        c'est la vérification qu'un chemin proposé est réellement réalisable.
        """
        return sum(self.edge_length(u, v) for u, v in zip(path, path[1:]))

    def straight_line_m(self, u: int, v: int) -> float:
        """Distance approchée à vol d'oiseau entre deux nœuds, en mètres."""
        return _distance_approx_m(*self.coords[u], *self.coords[v])

    def couts_reels_vers(self, goal: int) -> dict[int, float]:
        """
        Coût du MEILLEUR chemin réel de chaque nœud jusqu'à ``goal``, en mètres.

        C'est la quantité que toute heuristique cherche à estimer : le h*(n)
        de la théorie. Une heuristique est admissible si et seulement si
        h(n) <= h*(n) pour tout n, et c'est cette table qui permet de le
        vérifier arête par arête plutôt que de le supposer.

        Obtenu par Dijkstra sur le graphe INVERSÉ — un seul parcours donne la
        distance de tous les nœuds vers ``goal``, là où un parcours par nœud
        coûterait n fois plus cher. Les nœuds d'où ``goal`` est injoignable
        n'apparaissent pas dans le résultat.

        Réservé à la rétroaction : c'est un outil de mesure, pas une donnée
        transmise au solveur étudiant.
        """
        inverse = nx.DiGraph()
        inverse.add_nodes_from(self.coords)
        for (u, v), longueur in self._dist.items():
            inverse.add_edge(v, u, weight=longueur)
        return nx.single_source_dijkstra_path_length(inverse, goal, weight="weight")

    def aretes_minimales_vers(self, goal: int) -> dict[int, int]:
        """
        Nombre MINIMAL d'arêtes séparant chaque nœud de ``goal``.

        C'est la quantité que BFS minimise — des étapes, pas des mètres. Un
        BFS correct renvoie donc toujours un chemin comptant exactement ce
        nombre d'arêtes, même lorsqu'il est plus long en mètres que l'optimum.

        Obtenu comme ``couts_reels_vers`` par un seul parcours sur le graphe
        INVERSÉ, et réservé de la même façon à la rétroaction.
        """
        inverse = nx.DiGraph()
        inverse.add_nodes_from(self.coords)
        inverse.add_edges_from((v, u) for (u, v) in self._dist)
        return nx.single_source_shortest_path_length(inverse, goal)


def load_road_graph(path: str | Path = DEFAULT_GRAPH_PATH) -> RoadGraph:
    """
    Lit le fichier GraphML et en dérive les structures décrites plus haut.

    Deux rues distinctes peuvent relier les deux mêmes intersections. On ne
    conserve alors que la plus courte. Le solveur étudiant voit ainsi un
    simple graphe orienté valué.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Réseau introuvable : {path}\nGénérez-le avec :  python src/extract_data.py"
        )

    G = ox.load_graphml(path)

    # Réindexation : les identifiants OpenStreetMap (jusqu'à onze chiffres,
    # dispersés) sont remplacés par des numéros 0, 1, … n-1, attribués dans
    # l'ordre croissant des identifiants d'origine. Les étudiants manipulent
    # ainsi des nœuds lisibles ; la correspondance est conservée dans ``osm``
    # pour retrouver la rue réelle au moment de l'affichage.
    osm: dict[int, int] = dict(enumerate(sorted(int(n) for n in G.nodes)))
    numero: dict[int, int] = {identifiant: i for i, identifiant in osm.items()}

    coords: Coords = {
        numero[int(n)]: (float(d["y"]), float(d["x"])) for n, d in G.nodes(data=True)
    }

    # Coordonnées PLANES en mètres, dérivées des latitudes/longitudes par une
    # projection équirectangulaire locale centrée sur le réseau. C'est ce que
    # manipule le solveur étudiant : des mètres, jamais des degrés.
    #
    # x croît vers l'est, y vers le nord, l'origine étant au centre du réseau.
    # À l'échelle d'un quartier la déformation est de l'ordre du mètre, et le
    # facteur de sécurité garantit que la distance à vol d'oiseau ainsi
    # obtenue ne dépasse JAMAIS la longueur réelle d'une rue — condition de
    # l'admissibilité de l'heuristique.
    lat_centre = sum(lat for lat, _ in coords.values()) / len(coords)
    lon_centre = sum(lon for _, lon in coords.values()) / len(coords)
    echelle = FACTEUR_ADMISSIBILITE * EARTH_RADIUS_M
    cos_centre = math.cos(math.radians(lat_centre))
    coords_m: CoordsM = {
        n: (
            echelle * math.radians(lon - lon_centre) * cos_centre,
            echelle * math.radians(lat - lat_centre),
        )
        for n, (lat, lon) in coords.items()
    }

    dist: dict[tuple[int, int], float] = {}
    for u, v, d in G.edges(data=True):
        try:
            longueur = float(d["length"])
        except KeyError as exc:  # GraphML incomplet
            raise KeyError(
                f"Attribut d'arête manquant ({exc}). Régénérez le réseau avec "
                "python src/extract_data.py"
            ) from exc
        cle = (numero[int(u)], numero[int(v)])
        dist[cle] = min(longueur, dist.get(cle, math.inf))

    adj_dist: Adjacency = {n: [] for n in coords}
    for (u, v), longueur in dist.items():
        adj_dist[u].append((v, longueur))

    # Ordre stable : deux exécutions successives explorent le graphe
    # exactement de la même façon, donc les résultats sont reproductibles.
    for voisins in adj_dist.values():
        voisins.sort()

    return RoadGraph(
        adj_dist=adj_dist,
        coords_m=coords_m,
        coords=coords,
        osm=osm,
        nx_graph=G,
        _dist=dist,
    )
