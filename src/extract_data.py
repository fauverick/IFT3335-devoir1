"""
Extraction du sous-réseau routier depuis OpenStreetMap.

OUTIL ENSEIGNANT — Les étudiants n'ont pas à exécuter ce script : le fichier
``data/roads_map.graphml`` est déjà versionné dans le dépôt. Il est fourni
pour documenter la provenance exacte des données et permettre de régénérer
la carte (autre quartier, autre ville) sans refaire le travail.

Exemples
--------
    python src/extract_data.py
    python src/extract_data.py --place "Outremont, Montréal, Québec, Canada"
    python src/extract_data.py --point 45.5231,-73.5817 --dist 2500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import osmnx as ox

# La console Windows n'est pas toujours en UTF-8 : on évite ainsi un
# plantage d'affichage sur les accents.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

DEFAULT_PLACE = "Le Plateau-Mont-Royal, Montréal, Québec, Canada"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "roads_map.graphml"


def build_graph(
    place: str | None = DEFAULT_PLACE,
    point: tuple[float, float] | None = None,
    dist: int = 2000,
    network_type: str = "drive",
):
    """
    Télécharge le réseau routier et n'en conserve que la partie exploitable.

    Un seul traitement est essentiel pour le TP :
    ``largest_component(strongly=True)`` ne conserve que la plus grande
    composante fortement connexe. Tout couple (départ, arrivée) du graphe
    final est donc joignable, sens uniques compris — aucun étudiant ne se
    retrouve à déboguer un « chemin introuvable » qui n'est pas de sa faute.

    Le seul attribut d'arête que le TP exploite est ``length``, en mètres :
    c'est le coût g(n) que les algorithmes doivent minimiser. OSMnx le pose
    lui-même sur chaque arête, il n'y a donc rien à annoter.
    """
    if point is not None:
        G = ox.graph_from_point(point, dist=dist, network_type=network_type)
    else:
        G = ox.graph_from_place(place, network_type=network_type)

    return ox.truncate.largest_component(G, strongly=True)


def _parse_point(texte: str) -> tuple[float, float]:
    lat, lon = (float(x) for x in texte.split(","))
    return lat, lon


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    source = parseur.add_mutually_exclusive_group()
    source.add_argument("--place", default=DEFAULT_PLACE, help="lieu au sens de Nominatim")
    source.add_argument("--point", type=_parse_point, help="centre « lat,lon »")
    parseur.add_argument("--dist", type=int, default=2000, help="rayon en mètres (avec --point)")
    parseur.add_argument("--network-type", default="drive", help="drive, bike, walk, all…")
    parseur.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parseur.parse_args()

    cible = f"point {args.point} (r = {args.dist} m)" if args.point else args.place
    print(f"Téléchargement du réseau « {cible} » depuis OpenStreetMap…")
    G = build_graph(
        place=args.place,
        point=args.point,
        dist=args.dist,
        network_type=args.network_type,
    )

    longueurs = [float(d["length"]) for _, _, d in G.edges(data=True)]
    print(
        f"  {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes\n"
        f"  longueur par arête : {min(longueurs):.1f} à {max(longueurs):.1f} m"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=args.out)
    taille = args.out.stat().st_size / 1e6
    print(f"Écrit : {args.out} ({taille:.1f} Mo)")


if __name__ == "__main__":
    main()
