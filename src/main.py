"""
Point d'entrée du travail pratique : charge la carte, appelle VOS
algorithmes, vérifie les chemins obtenus, puis les affiche.

BOÎTE NOIRE — Ce fichier vous est fourni. Vous n'avez rien à y modifier ;
tout votre travail se fait dans ``student_solver.py``.

Exemples
--------
    python src/main.py                          # A*, chemin le plus court, animé
    python src/main.py --algo tous              # les quatre algorithmes comparés
    python src/main.py --algo bfs               # un seul algorithme, animé
    python src/main.py --seed 7                 # un autre trajet, au hasard
    python src/main.py --no-anim                # image fixe, plus rapide
    python src/main.py --save sortie/trajet.gif # enregistre l'animation
    python src/main.py --from 45.5231,-73.5817 --to 45.5270,-73.5950
    python src/main.py --algo tous --heuristique manhattan   # l'autre heuristique
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from road_graph import (
    DEFAULT_GRAPH_PATH,
    RoadGraph,
    SearchResult,
    format_cost,
    load_road_graph,
)

if hasattr(sys.stdout, "reconfigure"):  # console Windows non UTF-8
    sys.stdout.reconfigure(errors="replace")


# --------------------------------------------------------------- algorithmes
#: Ordre d'affichage : du plus naïf au plus informé.
ALGOS = ("bfs", "dfs", "ucs", "astar")

NOMS = {
    "bfs": "BFS",
    "dfs": "DFS",
    "ucs": "UCS (Dijkstra)",
    "astar": "A*",
}


# --------------------------------------------------------------------------- CLI
def _latlon(texte: str) -> tuple[float, float]:
    lat, lon = (float(x) for x in texte.split(","))
    return lat, lon


def analyser_arguments() -> argparse.Namespace:
    parseur = argparse.ArgumentParser(
        description="Recherche du meilleur itinéraire sur un réseau routier réel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parseur.add_argument("--algo", choices=[*ALGOS, "tous"], default="astar",
                         help="algorithme de recherche ; « tous » les compare "
                              "(défaut : astar)")
    parseur.add_argument("--heuristique", choices=("euclidienne", "manhattan"),
                         default="euclidienne",
                         help="heuristique employée par A* ; comparez les deux pour "
                              "voir laquelle est admissible (défaut : euclidienne)")
    parseur.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH,
                         help="fichier GraphML du réseau")
    parseur.add_argument("--start", type=int, help="numéro du nœud de départ (0..n-1)")
    parseur.add_argument("--goal", type=int, help="numéro du nœud d'arrivée (0..n-1)")
    parseur.add_argument("--from", dest="depart", type=_latlon, metavar="LAT,LON",
                         help="point de départ en coordonnées GPS")
    parseur.add_argument("--to", dest="arrivee", type=_latlon, metavar="LAT,LON",
                         help="point d'arrivée en coordonnées GPS")
    parseur.add_argument("--seed", type=int, default=0,
                         help="graine du trajet aléatoire (défaut : 0)")
    parseur.add_argument("--no-anim", action="store_true",
                         help="affiche une image fixe au lieu de l'animation")
    parseur.add_argument("--save", type=Path, metavar="FICHIER",
                         help="enregistre le résultat (.gif animé ou .png fixe)")
    parseur.add_argument("--no-plot", action="store_true",
                         help="n'affiche rien : uniquement le résumé texte")
    return parseur.parse_args()


# ------------------------------------------------------------ choix du trajet
def choisir_trajet(rg: RoadGraph, args: argparse.Namespace) -> tuple[int, int]:
    """Détermine (start, goal) selon les options, ou tire un long trajet."""
    if args.depart is not None or args.arrivee is not None:
        if args.depart is None or args.arrivee is None:
            raise SystemExit("Erreur : --from et --to vont ensemble.")
        return rg.nearest_node(*args.depart), rg.nearest_node(*args.arrivee)

    if (args.start is None) != (args.goal is None):
        raise SystemExit("Erreur : --start et --goal vont ensemble.")

    if args.start is not None:
        for nom, n in (("--start", args.start), ("--goal", args.goal)):
            if n not in rg.coords:
                raise SystemExit(
                    f"Erreur : le nœud {n} ({nom}) n'existe pas — "
                    f"les nœuds sont numérotés de 0 à {len(rg.coords) - 1}."
                )
        return args.start, args.goal

    # Trajet par défaut : la plus longue de quelques paires tirées au hasard,
    # pour que l'exploration soit visible. Le tirage dépend de --seed, donc
    # deux exécutions avec la même graine donnent le même trajet.
    tirage = random.Random(args.seed)
    noeuds = sorted(rg.coords)
    paires = ((tirage.choice(noeuds), tirage.choice(noeuds)) for _ in range(200))
    return max(paires, key=lambda p: rg.straight_line_m(*p))


# --------------------------------------------------------------- exécution
def executer(
    algo: str, rg: RoadGraph, start: int, goal: int,
    heuristique: str = "euclidienne",
) -> SearchResult:
    """
    Lance un algorithme du solveur étudiant et renvoie son résultat.

    Ce programme MESURE, il ne corrige pas : il rapporte le nombre de nœuds
    développés et le coût du chemin obtenu, sans jamais dire si ce chemin est
    le bon. C'est à vous de comparer les algorithmes entre eux et d'en tirer
    vos conclusions.

    Laisse remonter ``NotImplementedError`` : c'est à l'appelant de décider
    s'il s'agit d'une erreur fatale ou d'une simple case « à faire ».
    """
    import student_solver as ss  # importé ici pour un message d'erreur clair

    adj = rg.adj_dist
    if algo == "astar":
        # Les deux heuristiques de l'étudiant s'expriment en MÈTRES, comme les
        # poids de « adj » : elles s'emploient donc telles quelles.
        estimer = (ss.heuristic_euclidean if heuristique == "euclidienne"
                   else ss.heuristic_manhattan)
        h = lambda n: estimer(n, goal, rg.coords_m)                    # noqa: E731
        appel = lambda: ss.a_star(adj, start, goal, h)                 # noqa: E731
    else:
        recherche = {"bfs": ss.bfs, "dfs": ss.dfs, "ucs": ss.ucs}[algo]
        appel = lambda: recherche(adj, start, goal)                   # noqa: E731

    return appel()


# --------------------------------------------------------------- présentation
def resumer_itineraire(rg: RoadGraph, path: list[int], maximum: int = 12) -> list[str]:
    """Regroupe le chemin en segments de rue consécutifs portant le même nom."""
    etapes: list[list] = []
    for u, v in zip(path, path[1:]):
        nom = rg.edge_data(u, v).get("name", "voie sans nom")
        if isinstance(nom, list):
            nom = nom[0]
        metres = rg.edge_length(u, v)
        if etapes and etapes[-1][0] == nom:
            etapes[-1][1] += metres
        else:
            etapes.append([nom, metres])

    lignes = [f"    {nom} — {metres:.0f} m" for nom, metres in etapes[:maximum]]
    if len(etapes) > maximum:
        lignes.append(f"    … et {len(etapes) - maximum} autre(s) segment(s)")
    return lignes


# --------------------------------------------------- un seul algorithme
def executer_un(
    rg: RoadGraph, args: argparse.Namespace, start: int, goal: int
) -> int:
    algo = args.algo

    try:
        resultat = executer(algo, rg, start, goal, args.heuristique)
    except NotImplementedError as exc:
        sys.stdout.flush()
        print(
            f"\n{exc}\n\n"
            "Complétez les fonctions marquées « TODO » dans src/student_solver.py,\n"
            "puis relancez :  python src/main.py --algo tous --no-plot",
            file=sys.stderr,
        )
        return 1

    if not isinstance(resultat, SearchResult):
        print(
            f"\nErreur : {algo} doit renvoyer un SearchResult, "
            f"pas {type(resultat).__name__}.",
            file=sys.stderr,
        )
        return 1

    # On RAPPORTE ce que l'algorithme a produit — nœuds développés et coût du
    # chemin — sans dire si c'est le bon résultat. La comparaison, c'est
    # « --algo tous », et la conclusion vous appartient.
    print(f"Nœuds développés : {resultat.n_explored}")
    if resultat.found:
        print(f"Coût du chemin trouvé : {format_cost(resultat.cost)}")
        print(f"Longueur du chemin : {len(resultat.path)} nœuds "
              f"({len(resultat.path) - 1} arêtes)")
    else:
        print("Coût du chemin trouvé : aucun chemin trouvé")

    if resultat.found:
        # Le chemin, rue par rue : c'est ce qui rattache les nombres ci-dessus
        # à un trajet réel sur la carte.
        try:
            lignes = resumer_itineraire(rg, resultat.path)
        except (ValueError, KeyError, IndexError) as exc:
            print(f"\n  Itinéraire non affichable : {exc}")
        else:
            print("\n  itinéraire :")
            for ligne in lignes:
                print(ligne)

    if args.no_plot:
        return 0

    import visualizer

    titre = f"{NOMS[algo]} — chemin le plus court · nœud {start} vers nœud {goal}"
    anime = not args.no_anim and (args.save is None or args.save.suffix.lower() == ".gif")
    if anime:
        visualizer.animer_recherche(rg, resultat, start, goal, titre, args.save)
    else:
        visualizer.afficher_resultat(rg, resultat, start, goal, titre, args.save)
    return 0


# --------------------------------------------------- les quatre, comparés
def afficher_tableau(
    resultats: dict[str, SearchResult | None],
    heuristique: str = "euclidienne",
) -> None:
    """
    Tableau comparatif : les MESURES brutes des quatre algorithmes.

    Deux colonnes seulement portent le propos du travail — le nombre de nœuds
    développés et le coût du chemin obtenu. Aucun verdict n'est rendu : le
    programme ne désigne pas le bon résultat, il aligne les chiffres. C'est en
    les comparant entre eux, et en relançant avec l'autre heuristique, que
    vous tirez vos propres conclusions.
    """
    print(f"\n  {'Algorithme':<16}{'Explorés':>9}{'Arêtes':>8}{'Coût':>12}")
    print("  " + "-" * 45)

    for algo in ALGOS:
        resultat = resultats[algo]
        nom = NOMS[algo]
        if resultat is None:
            print(f"  {nom:<16}{'à faire':>9}")
            continue
        if not resultat.found:
            print(f"  {nom:<16}{resultat.n_explored:>9}"
                  f"{'aucun chemin trouvé':>20}")
            continue

        print(f"  {nom:<16}{resultat.n_explored:>9}{len(resultat.path) - 1:>8}"
              f"{f'{resultat.cost:.0f} m':>12}")

    autre = "manhattan" if heuristique == "euclidienne" else "euclidienne"
    print(f"\n  Relancez avec --heuristique {autre} et comparez les colonnes "
          f"« Explorés » et « Coût ».")


def comparer_tous(
    rg: RoadGraph, args: argparse.Namespace, start: int, goal: int
) -> int:
    resultats: dict[str, SearchResult | None] = {}
    for algo in ALGOS:
        try:
            resultat = executer(algo, rg, start, goal, args.heuristique)
        except NotImplementedError:
            resultats[algo] = None
            continue
        if not isinstance(resultat, SearchResult):
            print(f"Erreur : {algo} doit renvoyer un SearchResult, "
                  f"pas {type(resultat).__name__}.", file=sys.stderr)
            resultats[algo] = None
            continue
        resultats[algo] = resultat

    faits = {a: r for a, r in resultats.items() if r is not None}
    if not faits:
        print(
            "\nAucun des quatre algorithmes n'est implémenté.\n"
            "Complétez les fonctions marquées « TODO » dans src/student_solver.py,\n"
            "puis relancez :  python src/main.py --algo tous --no-plot",
            file=sys.stderr,
        )
        return 1

    afficher_tableau(resultats, args.heuristique)

    if args.no_plot:
        return 0

    import visualizer

    sortie = args.save
    if sortie is not None and sortie.suffix.lower() == ".gif":
        # La comparaison est une image fixe : quatre animations simultanées
        # ne se lisent pas.
        sortie = sortie.with_suffix(".png")
        print(f"\n  (comparaison enregistrée en image fixe : {sortie})")

    titre = (f"Quatre algorithmes — chemin le plus court · "
             f"nœud {start} vers nœud {goal}")
    visualizer.comparer_algorithmes(
        rg, {NOMS[a]: r for a, r in faits.items()}, start, goal, titre, sortie
    )
    return 0


# ----------------------------------------------------------------------- main
def main() -> int:
    args = analyser_arguments()

    rg = load_road_graph(args.graph)
    start, goal = choisir_trajet(rg, args)

    if args.algo == "tous":
        return comparer_tous(rg, args, start, goal)
    return executer_un(rg, args, start, goal)


if __name__ == "__main__":
    raise SystemExit(main())
