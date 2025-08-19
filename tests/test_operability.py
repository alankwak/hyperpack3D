import re

import pytest

from hyperpack import HyperPack

DEFAULT_POTENTIAL_POINTS_STRATEGY = HyperPack.DEFAULT_POTENTIAL_POINTS_STRATEGY


@pytest.mark.parametrize(
    "containers,items,points_seq,obj_val",
    [
        (((2, 3, 1), (2, 2, 2)), ((2, 3, 1), (1, 1, 1)), ("A", "B"), 1.0875),
        (((2, 3, 1),), ((2, 3, 1),), ("A", "B"), 1),
        (((2, 4, 3),), ((2, 4, 2), (1, 2, 1)), ("A", "B", "C"), 0.7499259259259259),
        (((2, 3, 1), (3, 3, 1), (3, 3, 1)), ((2, 2, 1), (3, 3, 1), (2, 1, 1)), ("A", "B"), 2),
    ],
)
def test_calculate_util(containers, items, points_seq, obj_val):
    containers = {f"cont-{i}": {"W": c[0], "L": c[1], "H": c[2]} for i, c in enumerate(containers)}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)
    prob._potential_points_strategy = points_seq
    prob.solve(debug=True)
    assert obj_val == prob.calculate_obj_value()
    assert len(prob.solution) == len(containers)


def test_deepcopy():
    items = ((2, 3, 1), (12, 3, 4), (12, 14, 12), (1, 1, 2), (4, 6, 7), (7, 9, 5), (1, 2, 3))
    containers = {"cont-0": {"W": 55, "L": 55, "H": 55}}
    items = {f"i-{i}": {"w": w, "l": l, "h": h} for i, (w, l, h) in enumerate(items)}
    prob = HyperPack(containers=containers, items=items)

    items_copy = prob.items.deepcopy()

    assert id(items_copy) != prob.items
    assert items_copy == prob.items
    prob.solve()
    solution_copy = prob._deepcopy_solution()
    assert id(solution_copy) != id(prob.solution)
    assert solution_copy == prob.solution
    obj_val_per_cont_copy = prob._copy_objective_val_per_container()
    assert id(obj_val_per_cont_copy) != id(prob.obj_val_per_container)
    assert obj_val_per_cont_copy == prob.obj_val_per_container
