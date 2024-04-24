from molecule import (
    get_all_angles,
    get_all_bonds,
    get_all_dihedrals,
    get_all_pairs_bond_separation,
)

from collections import Counter

import pytest


@pytest.fixture
def graph():
    return [[1, 5], [0, 2], [1, 3, 4], [2], [2], [0], [7], [6]]


def test_get_all_pairs_bond_separation(graph):
    inf = float("inf")
    assert get_all_pairs_bond_separation(graph) == [
        [0, 1, 2, 3, 3, 1, inf, inf],
        [1, 0, 1, 2, 2, 2, inf, inf],
        [2, 1, 0, 1, 1, 3, inf, inf],
        [3, 2, 1, 0, 2, 4, inf, inf],
        [3, 2, 1, 2, 0, 4, inf, inf],
        [1, 2, 3, 4, 4, 0, inf, inf],
        [inf, inf, inf, inf, inf, inf, 0, 1],
        [inf, inf, inf, inf, inf, inf, 1, 0],
    ]


def test_get_all_bonds(graph):
    assert Counter(get_all_bonds(graph)) == Counter(
        [
            (0, 1),
            (0, 5),
            (1, 2),
            (2, 3),
            (2, 4),
            (6, 7),
        ]
    )


def test_get_all_angles(graph):
    assert Counter(get_all_angles(graph)) == Counter(
        [
            (1, 0, 5),
            (0, 1, 2),
            (1, 2, 3),
            (1, 2, 4),
            (3, 2, 4),
        ]
    )


def test_get_all_dihedrals(graph):
    assert Counter(get_all_dihedrals(graph)) == Counter(
        [
            (5, 0, 1, 2),
            (0, 1, 2, 3),
            (0, 1, 2, 4),
        ]
    )
