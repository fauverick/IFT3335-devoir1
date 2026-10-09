"""
============================================================================
 FICHIER À COMPLÉTER: c'est le SEUL fichier que vous modifiez et remettez.
============================================================================

Prénom, Nom et Matricule:

Hung, Nguyen (20246446)
Larue, Olivier ()

"""
from __future__ import annotations
from math import inf

import heapq
import math
from collections import deque

from road_graph import Adjacency, CoordsM, SearchResult

# ============================================================================
# Fonctions utilitaires déjà fournies que vous devrez utiliser dans votre code des implémentations
# des algorithmes de recherche. Ne les modifier pas.
# ============================================================================
def reconstruct_path(came_from: dict[int, int], start: int, goal: int) -> list[int]:
    """
    Cette méthode reconstruit le chemin complet trouvé par l'algorithme de recherche depuis le noeud de départ (start)
    jusqu'au noeud d'arrivée (goal) en utilisant la table de prédécesseurs (came_from) qui a été remplie par l'algorithme de recherche.
    Chemin complet de ``start`` à ``goal``, reconstruit à partir de la liste des noeuds explorés.

    Pour chacun des algorithmes de recherches, on stock le noeud par lequel on est arrivé à un noeud donné. Cette liste est nommé *came_from*.
    Cette fonction ne fait que inverser cette liste pour reconstruire le chemin depuis le départ jusqu'à l'arrivée.
    """
    chemin = [goal]
    while chemin[-1] != start:
        chemin.append(came_from[chemin[-1]])
    chemin.reverse()
    return chemin


def path_cost(adj: Adjacency, chemin: list[int]) -> float:
    """
    Calcule le cout total du chemin retrouvé par l'algorithme de recherche. Le coût total est la somme des poids des arêtes du chemin.
    Cette méthode fonctionne pour tous les algorithmes de recherche suivant.
    """
    total = 0.0
    for u, v in zip(chemin, chemin[1:]):
        poids = next((p for voisin, p in adj[u] if voisin == v), None)
        if poids is None:
            raise ValueError(f"L'arête {u} -> {v} n'existe pas dans le graphe.")
        total += poids
    return total


# ============================================================================
# TODO 1/6: parcours en largeur (BFS)
# ============================================================================
def bfs(adj: Adjacency, start: int, goal: int) -> SearchResult:
    """
    Implémenter l'algorithme de recherche: parcours en largeur (BFS) pour trouver le chemin entre les noeuds *start* et *goal* sur le graphe *adj*.

    adj: Le graphe de la carte sous forme de liste d'adjacence pondérée où chaque noeud est associé à une liste de tuples (voisin, poids). Le poids représente le coût de l'arête entre le noeud et son voisin.
    (Voir le fichier description du devoir *devoir1.pdf* pour plus de détails)
    start: L'identifiant du noeud de départ.
    goal: L'identifiant du noeud d'arrivée.

    Consignes à respecter:
    - Ne changer pas la signature de la fonction.
    - Vous devez utiliser les variables initialisées *came_from*, *expanded* et *n_explored* dont leur explication est détaillé ci-bas.
      Elles servent au fonctionnement de l'algorithme, à la visualisation et à la correction donc il faut les populer.
    - Vous pouvez (devez) ajouter d'autres variables locales que vous jugez nécessaires.
    - La valeur de retour doit être:
            chemin = reconstruct_path(came_from, start, goal) 
            cout = path_cost(adj, chemin)
            return SearchResult(chemin, cout, expanded, n_explored) 
    """


    came_from: dict[int, int] = {}   # Dictionnaire pour stocker le chemin parcouru pour chaque noeud visité. la clé est le noeud courant et la valeur est le noeud parent utilisé pour atteindre ce noeud.  
    expanded: list[int] = []         # Liste pour stocker les noeuds développés dans leur ordre d'exploration.       
    n_explored = 0                   # Compteur pour compter le nombre de noeuds développés. Il doit être incrémenté à chaque fois qu'un noeud est développé.  

    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        visited.add(node)
        expanded.append(node)
        n_explored += 1

        if(node == goal):
            chemin = reconstruct_path(came_from, start, goal)
            cout = path_cost(adj, chemin)
            return SearchResult(chemin, cout, expanded, n_explored)

        for v, dist in adj[node]:
            if(v not in visited):
                visited.add(v) # to prevent revisiting and re-add the same node
                queue.append(v)
                came_from[v] = node

    return SearchResult(None, math.inf, expanded, n_explored) # N'effacez pas cette ligne


# ============================================================================
# TODO 2/6: parcours en profondeur (DFS)
# ============================================================================
def dfs(adj: Adjacency, start: int, goal: int) -> SearchResult:
    """
    Implémenter l'algorithme de recherche: parcours en profondeur (DFS).

    Consignes à respecter: Même consignes que pour BFS.
    """
    came_from: dict[int, int] = {}
    expanded: list[int] = []
    n_explored = 0

    visited = set()
    stack = deque([start])

    while(stack):
        node = stack.pop()
        visited.add(node)
        expanded.append(node)
        n_explored += 1

        # print(node)

        if(node == goal):
            chemin = reconstruct_path(came_from, start, goal)
            cout = path_cost(adj, chemin)
            return SearchResult(chemin, cout, expanded, n_explored)

        for v, dist in (adj[node]):
            if(v not in visited):
                visited.add(v)
                stack.append(v)
                came_from[v] = node

    return SearchResult(None, math.inf, expanded, n_explored) # N'effacez pas cette ligne


# ============================================================================
# TODO 3/6: recherche à coût uniforme (UCS / Dijkstra)
# ============================================================================
def ucs(adj: Adjacency, start: int, goal: int) -> SearchResult:
    """
    Implémenter l'algorithme de recherche: recherche à coût uniforme (UCS / Dijkstra).

    Consignes à respecter: 
    - Même consignes que pour BFS.
    - N'oublier pas de gérer les doublons dans la file de priorité. Vous pouvez utiliser heapq.
    """
    came_from: dict[int, int] = {}
    expanded: list[int] = []
    n_explored = 0


    dist : dict[int, float] = {}

    for v in adj:
        dist[v] = float(inf)

    dist[start] # set start to 0

    pq = [(0, start)]
    heapq.heapify(pq)

    while(pq):
        current_dist, node = heapq.heappop(pq)
        if(current_dist > dist[node]):
            continue

        expanded.append(node)
        n_explored += 1

        if(node == goal):
            chemin = reconstruct_path(came_from, start, goal)
            cout = path_cost(adj, chemin)
            return SearchResult(chemin, cout, expanded, n_explored)

        for v, next_dist in adj[node]:
            new_dist = current_dist + next_dist
            if (new_dist < dist[v]):
                dist[v] = new_dist
                came_from[v] = node
                heapq.heappush(pq, (new_dist, v))

    return SearchResult(None, math.inf, expanded, n_explored) # N'effacez pas cette ligne


# ============================================================================
# TODO 4/6 : Heuristique euclidienne
# ============================================================================
def heuristic_euclidean(node: int, goal: int, coords_m: CoordsM) -> float:
    """
    Distance à vol d'oiseau, en MÈTRES : la longueur du segment de droite
    reliant *node*, le noeud courant à *goal*, le noeud d'arrivée.

    node: Identifiant du noeud courant.
    goal: Identifiant du noeud d'arrivée.
    coords_m: Dictionnaire {noeud: (x, y)} donnant les coordonnées en mètres de chaque noeud.

    Pour obtenir la position géographique d'un noeud (en mètres):
        x_noeud, y_noeud = coords_m[noeud]

    Pour caluler la distance euclidienne entre deux points (x1, y1) et (x2, y2), vous pouvez utiliser la fonction math.hypoth
    """

    x1, y1 = coords_m[node]
    x2, y2 = coords_m[goal]
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    return d

# ============================================================================
# TODO 5/6 : Heuristique manhattan
# ============================================================================
def heuristic_manhattan(node: int, goal: int, coords_m: CoordsM) -> float:
    """
    Distance de Manhattan, en MÈTRES : la somme des deux déplacements, l'un
    vers l'est, l'autre vers le nord, comme si l'on ne pouvait circuler que
    parallèlement aux axes.

    Consigne à respecter: Utiliser la fonction abs() 
    """

    x1, y1 = coords_m[node]
    x2, y2 = coords_m[goal]
    d = abs(x1 - x2) + abs(y1 - y2)
    return d


# ============================================================================
# TODO 6/6 — l'algorithme A*
# ============================================================================
def a_star(adj: Adjacency, start: int, goal: int, heuristic) -> SearchResult:
    """
    Implémenter l'algorithme de recherche: A*.

    heuristic: fonction à un seul argument (ID d'un noeud) qui renvoie une estimation du coût restant pour atteindre le but.
    (Soit manhattant ou euclidienne, c'est une abstraction pour implémenter A* agnostic à l'heuristique utilisée.)
    
    A* est définie selon la fonction f(n) = g(n) + h(n), donc vous pouvez simplement écrire h = heuristic(noeud) pour représenter h(n)

    Consignes à respecter: 
    - Même consignes générales que pour les autres algorithmes de recherche.
    - le coût renvoyé dans SerachResult doit être g[goal], pas f[goal], la somme du coût réel et de l'heurisitque, on veut le coût réel du chemin.
    - N'oubliez pas de gérer les doublons dans l'exploration d'un graphe.
    - Un noeud déjà développé (fermé) ne doit jamais être rouvert ni mis à jour: ignorez-le lorsqu'il apparaît
      comme voisin. Ainsi, le chemin renvoyé et son coût g[goal] concordent toujours
    - Vous pouvez utiliser *heapq* pour gérer la file de priorité.

    """
    came_from: dict[int, int] = {}
    expanded: list[int] = []
    n_explored = 0

    heap_start = [(0, start)]
    heapq.heapify(heap_start)
    closed = set()
    g: dict[int, float] = {} # actual cost to get to a node

    for u in adj:
        g[u] = float(inf)
    
    g[start] = 0

    while heap_start:
        _, node = heapq.heappop(heap_start)
        if node in closed: 
            continue
        
        closed.add(node)
        expanded.append(node)
        n_explored += 1
        if (node == goal):
            chemin = reconstruct_path(came_from, start, goal)
            cout = path_cost(adj, chemin)
            return SearchResult(chemin, cout, expanded, n_explored)
        
        for v, next_dist in adj[node]:
            if (v in closed):
                continue

            new_dist = g[node] + next_dist
            if (new_dist < g[v]):
                g[v] = new_dist
                came_from[v] = node
                f = g[v] + heuristic(v)  # estimated cost
                heapq.heappush(heap_start, (f, v))

    return SearchResult(None, math.inf, expanded, n_explored) # N'effacez pas cette ligne
