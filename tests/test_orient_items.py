import re

import pytest

from hyperpack import HyperPack

DEFAULT_POTENTIAL_POINTS_STRATEGY = HyperPack.DEFAULT_POTENTIAL_POINTS_STRATEGY


@pytest.mark.parametrize(
    "orientation",
    ["short", "tall"],
)
def test_orient_items(orientation, request):
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)
    items = prob._items.deepcopy()
    init_items = prob._items.deepcopy()

    return_value = prob.orient_items(orientation=orientation)
    assert return_value is None
    assert list(prob.items.items()) != list(init_items.items())
    for _, item in prob.items.items():
        if orientation == "tall":
            assert item["h"] >= item["l"] and item["h"] >= item["w"]
        else:
            assert item["h"] <= item["l"] and item["h"] <= item["w"]


def test_orient_items__no_rotation_warning(caplog):
    settings = {"rotation": False}
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items, settings=settings)
    return_value = prob.orient_items()
    assert items == prob.items
    assert "can't rotate items. Rotation is disabled" in caplog.text
    assert return_value is None


def test_orient_items__wrong_orientation_parameter(caplog):
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)
    orientation = "wrong_param"
    return_value = prob.orient_items(orientation=orientation)
    assert items == prob.items
    assert (
        f"orientation parameter '{orientation}' not valid. Orientation skipped."
        in caplog.text
    )
    assert return_value is None


def test_orient_items__orientation_None(caplog):
    items = ((2, 3, 2), (12, 3, 4), (12, 14, 9), (1, 1, 5), (4, 6, 12), (7, 9, 7), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)
    return_value = prob.orient_items(orientation=None)
    assert items == prob.items
    assert (
        f"orientation parameter '{None}' not valid. Orientation skipped."
        not in caplog.text
    )
    assert return_value is None
