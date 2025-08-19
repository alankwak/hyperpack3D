import re

import pytest

from hyperpack import HyperPack

DEFAULT_POTENTIAL_POINTS_STRATEGY = HyperPack.DEFAULT_POTENTIAL_POINTS_STRATEGY


@pytest.mark.parametrize(
    "sorting_by",
    [
        ("volume", True),
        ("surface_area", True),
        ("longest_side_ratio", True),
        ("volume", False),
        ("surface_area", False),
        ("longest_side_ratio", False),
        ("NotImplemented", None),
    ],
)
def test_sorting(sorting_by):
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)

    by, reverse = sorting_by
    init_items = prob._items.deepcopy(items)

    if by == "NotImplemented":
        with pytest.raises(NotImplementedError):
            prob.sort_items(sorting_by=sorting_by)
        return

    prob.sort_items(sorting_by=sorting_by)
    assert list(prob.items.items()) != list(init_items.items())
    first_item = list(prob.items.items())[0]
    if by == "volume":
        previous_quantity = first_item[1]["w"] * first_item[1]["l"] * first_item[1]["h"]
    elif by == "surface_area":
        previous_quantity = first_item[1]["l"] * first_item[1]["w"] * 2 + first_item[1]["l"] * first_item[1]["h"] * 2 + first_item[1]["w"] * first_item[1]["h"] * 2
    elif by == "longest_side_ratio":
        previous_quantity = max(first_item[1]["w"], first_item[1]["l"], first_item[1]["h"]) / min(
            first_item[1]["w"], first_item[1]["l"],first_item[1]["h"]
        )

    for _, item in list(prob.items.items())[1:]:
        if by == "volume":
            quantity = item["w"] * item["l"] * item["h"]
        elif by == "surface_area":
            quantity = item["l"] * item["w"] * 2 + item["l"] * item["h"] * 2 + item["w"] * item["h"] * 2
        elif by == "longest_side_ratio":
            quantity = max(item["w"], item["l"], item["h"]) / min(item["w"], item["l"], item["h"])

        if reverse:
            assert quantity <= previous_quantity
        else:
            assert quantity >= previous_quantity

        previous_quantity = quantity

    assert prob.items.__class__.__name__ == "Items"


def test_sorting_by_None(caplog):
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)

    ret = prob.sort_items(sorting_by=None)
    assert ret == None
