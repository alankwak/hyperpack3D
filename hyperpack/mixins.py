import math
import queue
import sys
import time
from .abstract import AbstractLocalSearch
from .loggers import hyperLogger, logger
from . import constants

from .exceptions import (
    SettingsError,
    FigureExportError,
)
from array import array
from collections import deque
from copy import deepcopy
from pathlib import Path
import re


class PointGenerationMixin:
    """
    Mixin providing the point generation functionality.

    Attributes required for solving operation:
        ``_containers`` attribute of type ``Containers``.
        ``_items`` attribute of type ``Items``.
        ``_potential_points_strategy`` strategy attribute set pre-solving.
        ``_rotate`` attribute for rotation.
        ``_strip_pack`` attribute.
        ``_container_height`` attribute.
    """

    # solving constants
    DEFAULT_POTENTIAL_POINTS_STRATEGY = (
        "A_z",
        "B_z",
        "A",
        "B",
        "C",
    )
    INIT_POTENTIAL_POINTS = {
        "O": (0, 0, 0),
        "A": deque(),
        "A_z": deque(),
        "B": deque(),
        "B_z": deque(),
        "C": deque(),
    }

    # % --------- construction heuristic methods ----------

    def _check_3d_fitting(self, W, L, H, Xo, Yo, Zo, w, l, h, current_items, requires_support=False,  xy_planes=None) -> bool:
        """
        Checks if the item with coordinates (Xo, Yo, Zo)
        and dimensions (w, l, h) fits without colliding with any
        of the items currently in the container (current_items).

        Optionally, setting ``requires_support`` to True checks if 
        the item's bottom plane has at least ``support_ratio`` support 
        from items beneath it. ``xy_planes`` is mandatory in this case
        and holds all planes at the z-level of the item's bottom plane.
        """

        if requires_support and xy_planes is None:
            raise SettingsError(
                "xy_planes must be provided when requires_support is True"
            )

        # Check if the item fits within the container dimensions
        if(Xo + w > W or Yo + l > L or Zo + h > H):
            return False

        # Check for collisions with existing items
        for item_id, item in current_items.items():
            X, Y, Z = item["Xo"], item["Yo"], item["Zo"]
            w_, l_, h_ = item["w"], item["l"], item["h"]

            if (
                Xo + w > X and X + w_ > Xo and
                Yo + l > Y and Y + l_ > Yo and
                Zo + h > Z and Z + h_ > Zo
            ):
                return False
            
        if requires_support:
            support_ratio = self._settings.get("support_ratio", 0.55)
            percent_support_z = 0
            bottom_area = w * l
            for plane in xy_planes.get(Zo, []):
                intersecting_area = max(0, min(Xo + w, plane[1][0]) - max(Xo, plane[0][0])) * max(0, min(Yo + l, plane[1][1]) - max(Yo, plane[0][1]))
                percent_support_z += intersecting_area / bottom_area
            
            if percent_support_z < support_ratio:
                return False

        return True

    def _generate_3d_points(
            self, container_items, outer_xy_planes, outer_xz_planes, outer_yz_planes, inner_xy_planes, inner_xz_planes, inner_yz_planes, debug=False
        ):
        """
        Generates a rich set of potential points from the corners of the newly placed item,
        including projections and points in concave corners.
        """

        potential_points = {
            "A": deque(),
            "B": deque(),
            "A_z": deque(),
            "B_z": deque(),
            "C": deque(),
            "A_x": deque(),
            "B_y": deque(),
            "C_xy": deque(),
        }

        for item_id, item in sorted(container_items.items()):
            Xo, Yo, Zo, w, l, h = item["Xo"], item["Yo"], item["Zo"], item["w"], item["l"], item["h"]
            
            x, y, z = 0, 1, 2
            # Define the 3 corners of the item
            A = (Xo, Yo + l, Zo) # origin + y
            B = (Xo + w, Yo, Zo) # origin + x
            C = (Xo, Yo, Zo + h) # origin + z

            blockC = False
            for plane in inner_xy_planes.get(Zo + h, []):
                if plane[0][0] <= C[x] < plane[1][0] and plane[0][1] <= C[y] < plane[1][1]:
                    blockC = True
                    break
                    
            blockA = False
            blockA_z = False
            for plane in inner_xz_planes.get(Yo + l, []):
                if plane[0][0] <= A[x] < plane[1][0] and plane[0][1] <= A[z] < plane[1][1]:
                    blockA = True
                if plane[0][0] <= A[x] < plane[1][0] and plane[0][1] <= A[z] + h < plane[1][1]:
                    blockA_z = True

            blockB = False
            blockB_z = False
            for plane in inner_yz_planes.get(Xo + w, []):
                if plane[0][0] <= B[y] < plane[1][0] and plane[0][1] <= B[z] < plane[1][1]:
                    blockB = True
                if plane[0][0] <= B[y] < plane[1][0] and plane[0][1] <= B[z] + h < plane[1][1]:
                    blockB_z = True
            
            # --------- add C, C'x, C'y points if possible ---------
            blockC_x = False
            blockC_y = False
            if not blockC:
                # check if C' can be projected to C'x
                support_x = 0
                for x_level in sorted(outer_yz_planes.keys(), reverse=True):
                    
                    if x_level <= C[x]:
                        for plane in outer_yz_planes[x_level]:
                            if x_level == C[x] and plane[0][0] <= C[y] < plane[1][0] and plane[0][1] <= C[z] < plane[1][1]:
                                blockC_x = True
                                break
                            elif plane[0][0] <= C[y] < plane[1][0] and plane[0][1] <= C[z] < plane[1][1]:
                                support_x = x_level
                                break
                    
                    if support_x > 0 or blockC_x: break
                
                if not blockC_x:
                    potential_points["C"].append((support_x, C[y], C[z]))
                if debug:
                    logger.debug(
                        f"\tProjected C ({C[x]}, {C[y]}, {C[z]}) to C'x ({support_x}, {C[y]}, {C[z]})")
                
                # check if C' can be projected to C'y
                support_y = 0
                for y_level in sorted(outer_xz_planes.keys(), reverse=True):
                    
                    if y_level <= C[y]:
                        for plane in outer_xz_planes[y_level]:
                            if y_level == C[y] and plane[0][0] <= C[x] < plane[1][0] and plane[0][1] <= C[z] < plane[1][1]:
                                blockC_y = True
                                break
                            elif plane[0][0] <= C[x] < plane[1][0] and plane[0][1] <= C[z] < plane[1][1]:
                                support_y = y_level
                                break
                    
                    if support_y > 0 or blockC_y: break
                
                if not blockC_y:
                    potential_points["C"].append((C[x], support_y, C[z]))
                if debug:
                    logger.debug(
                        f"\tProjected C ({C[z]}, {C[z]}, {C[z]}) to C'y ({C[x]}, {support_y}, {C[z]})")
                
                # add C point
                potential_points["C"].append(C)
                if debug:
                    logger.debug(f"\tAdded C point: {C}")
            
            # --------- add A, A'z, A'x points if possible ---------
            if not blockA:
                support_x = 0
                for x_level in sorted(outer_yz_planes.keys(), reverse=True):
                    if x_level <= A[y]:
                        for plane in outer_yz_planes[x_level]:
                            if plane[0][0] <= A[y] < plane[1][0] and plane[0][1] <= A[z] < plane[1][1]:
                                support_x = x_level
                                break
                        if support_x > 0: break
                if support_x != Xo:
                    potential_points["A"].append((support_x, A[y], A[z]))
                    if debug:
                        logger.debug(f"\t Projected A ({A[x]}, {A[y]}, {A[z]}) to A'x ({support_x}, {A[y]}, {A[z]})")
                
                potential_points["A"].append(A)
                if debug:
                    logger.debug(f"\tAdded A point: {A}")
            
            if not blockA_z:
                support_z = 0
                for z_level in sorted(outer_xy_planes.keys(), reverse=True):
                    if z_level <= A[z] + h:
                        for plane in outer_xy_planes[z_level]:
                            if plane[0][0] <= A[x] < plane[1][0] and plane[0][1] <= A[y] < plane[1][1] - 5:
                                support_z = z_level
                                break
                            if z_level != A[z] + h and plane[0][1] <= A[y] < plane[1][1] and plane[0][0] <= A[x] + w and plane[1][0] > A[x]:
                                potential_points["A_z"].append((plane[0][0], A[y], z_level))
                                if debug:
                                    logger.debug(f"\t Projected A ({A[x]}, {A[y]}, {A[z]}) to A'z ({A[x]}, {A[y]}, {z_level}) using edge")
                        if support_z > 0: break
                if support_z != Zo:
                    potential_points["A_z"].append((A[x], A[y], support_z))
                    if debug:
                        logger.debug(f"\t Projected A ({A[x]}, {A[y]}, {A[z]}) to A'z ({A[x]}, {A[y]}, {support_z})")

            # --------- add B, B'z, B'y points if possible ---------
            if not blockB:
                support_y = 0
                for y_level in sorted(outer_xz_planes.keys(), reverse=True):
                    if y_level <= B[y]:
                        for plane in outer_xz_planes[y_level]:
                            if plane[0][0] <= B[x] < plane[1][0] and plane[0][1] <= B[z] < plane[1][1]:
                                support_y = y_level
                                break
                        if support_y > 0: break
                if support_y != Yo:
                    potential_points["B"].append((B[x], support_y, B[z]))
                    if debug:
                        logger.debug(f"\t Projected B ({B[x]}, {B[y]}, {B[z]}) to B'y ({B[x]}, {support_y}, {B[z]})")
                
                potential_points["B"].append(B)
                if debug:
                    logger.debug(f"\tAdded B point: {B}")
                
            if not blockB_z:
                support_z = 0
                for z_level in sorted(outer_xy_planes.keys(), reverse=True):
                    if z_level < B[z] + h:
                        for plane in outer_xy_planes[z_level]:
                            if plane[0][0] <= B[x] < plane[1][0] and plane[0][1] <= B[y] < plane[1][1] - 5:
                                support_z = z_level
                                break
                            if plane[0][0] <= B[x] < plane[1][0] and plane[0][1] <= B[y] + l and plane[1][1] > B[y]:
                                potential_points["B_z"].append((B[x], plane[0][1], z_level))
                                if debug:
                                    logger.debug(f"\t Projected B ({B[x]}, {B[y]}, {B[z]}) to B'z ({B[x]}, {B[y]}, {z_level}) using edge")
                        if support_z > 0: break
                if support_z != Zo:
                    potential_points["B_z"].append((B[x], B[y], support_z))
                    if debug:
                        logger.debug(f"\t Projected B ({B[x]}, {B[y]}, {B[z]}) to B'z ({B[x]}, {B[y]}, {support_z})")
                        
        return potential_points

    def _append_outer_planes(self, xy_planes, xz_planes, yz_planes, Xo, Yo, Zo, w, l, h):
        # xy plane
        if Zo + h in xy_planes:
            xy_planes[Zo + h].append(((Xo, Yo), (Xo + w, Yo + l)))
        else:
            xy_planes[Zo + h] = [((Xo, Yo), (Xo + w, Yo + l))]

        # xz plane
        if Yo + l in xz_planes:
            xz_planes[Yo + l].append(((Xo, Zo), (Xo + w, Zo + h)))
        else:
            xz_planes[Yo + l] = [((Xo, Zo), (Xo + w, Zo + h))]

        # yz plane
        if Xo + w in yz_planes:
            yz_planes[Xo + w].append(((Yo, Zo), (Yo + l, Zo + h)))
        else:
            yz_planes[Xo + w] = [((Yo, Zo), (Yo + l, Zo + h))]
    
    def _append_inner_planes(self, xy_planes, xz_planes, yz_planes, Xo, Yo, Zo, w, l, h):
        # xy plane
        if Zo in xy_planes:
            xy_planes[Zo].append(((Xo, Yo), (Xo + w, Yo + l)))
        else:
            xy_planes[Zo] = [((Xo, Yo), (Xo + w, Yo + l))]

        # xz plane
        if Yo in xz_planes:
            xz_planes[Yo].append(((Xo, Zo), (Xo + w, Zo + h)))
        else:
            xz_planes[Yo] = [((Xo, Zo), (Xo + w, Zo + h))]

        # yz plane
        if Xo in yz_planes:
            yz_planes[Xo].append(((Yo, Zo), (Yo + l, Zo + h)))
        else:
            yz_planes[Xo] = [((Yo, Zo), (Yo + l, Zo + h))]

    def _get_initial_container_height(self, container):
        if self._strip_pack:
            return self._container_height
        else:
            return container["H"]

    def _get_initial_potential_points(self):
        return {
            "O": (0, 0, 0),
            "A": deque(),
            "B": deque(),
            "A_z": deque(),
            "A_x": deque(),
            "B_z": deque(),
            "B_y": deque(),
            "C": deque(),
            "C_xy": deque(),
        }

    def _get_initial_outer_xy_planes(self, W, L, H):
        return {0: [((0, 0), (W, L))]}
    
    def _get_initial_outer_xz_planes(self, W, L, H):
        return {0: [((0, 0), (W, H))]}
    
    def _get_initial_outer_yz_planes(self, W, L, H):
        return {0: [((0, 0), (L, H))]}

    def _get_initial_point(self, potential_points, **kwargs):
        return potential_points["O"], "O"

    def calculate_objective_value(
        self, obj_value, w, l, h, W, L, H
    ):
        return obj_value + (w * l * h) / (W * L * H)

    # In mixins.py, add this new method inside PointGenerationMixin
    def _find_placement_in_container(self, item, item_id, container, container_state):
        """
        Tries to find a valid placement for a SINGLE item in a given container.
        
        Args:
            item (dict): The item to place.
            item_id (str): The ID of the item.
            container (dict): The container's dimensions.
            container_state (dict): The current state of the container, including
                                    placed_items, potential_points, and planes.

        Returns:
            A tuple (found_placement, new_container_state) where found_placement
            is a boolean and new_container_state is the updated state if found.
        """
        W, L, H = container["W"], container["L"], self._get_initial_container_height(container)
        
        # Create a copy of the state to modify, so we don't alter the original if placement fails
        temp_state = deepcopy(container_state)
        potential_points = temp_state['potential_points']

        # We need to check every potential point, not just the first one
        points_to_try = []
        for pclass in self._potential_points_strategy:
            points_to_try.extend(potential_points[pclass])
        
        # Add the origin as a fallback if no other points exist
        if not points_to_try:
            points_to_try.append((0,0,0))

        for point in points_to_try:
            Xo, Yo, Zo = point
            
            if self._rotation:
                # Try all 6 orientations
                for rotation_state in range(2 if item.get('horizontal_rotation_only', False) else 6):
                    w, l, h = item['w'], item['l'], item['h']
                    if rotation_state == 1: w, l, h = l, w, h
                    elif rotation_state == 2: w, l, h = w, h, l
                    elif rotation_state == 3: w, l, h = h, w, l
                    elif rotation_state == 4: w, l, h = l, h, w
                    elif rotation_state == 5: w, l, h = h, l, w

                    # Use the 3D fitting check
                    if self._check_3d_fitting(W, L, H, Xo, Yo, Zo, w, l, h, temp_state['placed_items'], requires_support=True, xy_planes=temp_state['outer_xy_planes']):
                        # --- Success! A placement was found. ---

                        # Update the item with its position and add it to the state
                        placed_item = item.copy()
                        placed_item.update({"Xo": Xo, "Yo": Yo, "Zo": Zo, "w": w, "l": l, "h": h})
                        temp_state['placed_items'][item_id] = placed_item
                        
                        # Generate new points and planes based on this placement
                        self._append_outer_planes(temp_state['outer_xy_planes'], temp_state['outer_xz_planes'], temp_state['outer_yz_planes'], Xo, Yo, Zo, w, l, h)
                        self._append_inner_planes(temp_state['inner_xy_planes'], temp_state['inner_xz_planes'], temp_state['inner_yz_planes'], Xo, Yo, Zo, w, l, h)
                        temp_state["potential_points"] = self._generate_3d_points(temp_state["placed_items"], 
                            temp_state['outer_xy_planes'], temp_state['outer_xz_planes'], temp_state['outer_yz_planes'],
                            temp_state['inner_xy_planes'], temp_state['inner_xz_planes'], temp_state['inner_yz_planes']
                        ) 
                        
                        return True, temp_state # Return success and the new state
            else:
                w, l, h = item['w'], item['l'], item['h']

                # Use the 3D fitting check
                if self._check_3d_fitting(W, L, H, Xo, Yo, Zo, w, l, h, temp_state['placed_items'], requires_support=True, xy_planes=temp_state['outer_xy_planes']):
                    # --- Success! A placement was found. ---

                    # Update the item with its position and add it to the state
                    placed_item = item.copy()
                    placed_item.update({"Xo": Xo, "Yo": Yo, "Zo": Zo, "w": w, "l": l, "h": h})
                    temp_state['placed_items'][item_id] = placed_item
                    
                    # Generate new points and planes based on this placement
                    self._append_outer_planes(temp_state['outer_xy_planes'], temp_state['outer_xz_planes'], temp_state['outer_yz_planes'], Xo, Yo, Zo, w, l, h)
                    self._append_inner_planes(temp_state['inner_xy_planes'], temp_state['inner_xz_planes'], temp_state['inner_yz_planes'], Xo, Yo, Zo, w, l, h)
                    temp_state["potential_points"] = self._generate_3d_points(temp_state["placed_items"], 
                        temp_state['outer_xy_planes'], temp_state['outer_xz_planes'], temp_state['outer_yz_planes'],
                        temp_state['inner_xy_planes'], temp_state['inner_xz_planes'], temp_state['inner_yz_planes']
                    ) 
                    
                    return True, temp_state # Return success and the new state

        # If we get here, no placement was found for this item in this container
        return False, container_state

    def _get_container_solution(self, current_solution):
        """
        Returns the solution object of the _construct method
        for the current solving container.
        """
        solution = {}
        for _id, item in current_solution.items():
            
            solution[_id] = [
                item["Xo"],
                item["Yo"],
                item["Zo"],
                item["w"],
                item["l"],
                item["h"]
            ]
        return solution

    def _solve(self, sequence=None, debug=False) -> None:
        """
        Solves for all containers using a First Fit heuristic.
        For each item, it tries to place it in the first container where it fits.
        """
        if sequence is None:
            items = self._items.deepcopy()
        else:
            items = self._items.deepcopy(sequence)

        # Initialize states for all containers
        container_states = {}
        for cont_id in self._containers:
            W = self._containers[cont_id]['W']
            L = self._containers[cont_id]['L']
            H = self._containers[cont_id]['H']
            container_states[cont_id] = {
                'placed_items': {},
                'potential_points': self._get_initial_potential_points(),
                'outer_xy_planes': self._get_initial_outer_xy_planes(W, L, H),
                'outer_xz_planes': self._get_initial_outer_xz_planes(W, L, H),
                'outer_yz_planes': self._get_initial_outer_yz_planes(W, L, H),
                'inner_xy_planes': {},
                'inner_xz_planes': {},
                'inner_yz_planes': {},
            }

        # Main First Fit Loop: Iterate through ITEMS first
        for item_id, item in items.items():
            # Then, for each item, iterate through CONTAINERS
            for cont_id in self._containers:
                
                # Try to place the current item in the current container
                was_placed, new_state = self._find_placement_in_container(
                    item, item_id, self._containers[cont_id], container_states[cont_id]
                )
                if was_placed:
                    # If placed, update the container's state and move to the NEXT ITEM
                    container_states[cont_id] = new_state
                    break # Stop searching for a container for this item
        
        # Post-processing: Convert final states into the required solution format
        solution = {}
        obj_val_per_container = {}
        for cont_id in self._containers:
            placed_items_in_cont = container_states[cont_id]['placed_items']
            solution[cont_id] = self._get_container_solution(placed_items_in_cont)

            # Calculate final utilization for the objective function
            total_items_vol = sum(d['w'] * d['l'] * d['h'] for d in placed_items_in_cont.values())
            total_cont_vol = self._containers[cont_id]['W'] * self._containers[cont_id]['L'] * self._containers[cont_id]['H']
            obj_val_per_container[cont_id] = total_items_vol / total_cont_vol if total_cont_vol > 0 else 0

        return solution, obj_val_per_container

class SolutionLoggingMixin:
    """
    Mixin for logging the solution.
    """

    def log_solution(self) -> str:
        """
        Logs the solution.

        If a solution isn't available a proper message is displayed.
        """
        if not getattr(self, "solution", False):
            hyperLogger.warning("No solving operation has been concluded.")
            return

        log = ["\nSolution Log:"]
        percent_items_stored = sum(
            [len(i) for cont_id, i in self.solution.items()]
        ) / len(self._items)
        log.append(f"Percent total items stored : {percent_items_stored*100:.4f}%")

        for cont_id in self._containers:
            L = self._containers[cont_id]["L"]
            W = self._containers[cont_id]["W"]
            H = self._containers._get_height(cont_id)
            log.append(f"Container: {cont_id} {W}x{L}x{H}")
            total_items_volume = sum(
                [i[3] * i[4] * i[5] for _, i in self.solution[cont_id].items()]
            )
            log.append(f"\t[util%] : {total_items_volume*100/(W*L*H):.4f}%")
            if self._strip_pack:
                solution = self.solution[cont_id]
                # height of items stack in solution
                max_height = max(
                    [solution[item_id][2] + solution[item_id][5] for item_id in solution] # Zo + h = total height
                    or [0]
                )
                log.append(f"\t[max height] : {max_height}")

        items_ids = {_id for cont_id, items in self.solution.items() for _id in items}
        remaining_items = [_id for _id in self._items if _id not in items_ids]
        log.append(f"\nRemaining items : {remaining_items}")
        output_log = "\n".join(log)
        hyperLogger.info(output_log)
        return output_log


class SolutionFigureMixin:
    """
    Mixin implementing the methods for building the
    figures of the solution.

    Extends the settings validation to support the
    figure operation.

    Must be used on leftmost position in the inheritance.
    """

    FIGURE_FILE_NAME_REGEX = re.compile(r"[a-zA-Z0-9_-]{1,45}$")
    FIGURE_DEFAULT_FILE_NAME = "PlotlyGraph"
    ACCEPTED_IMAGE_EXPORT_FORMATS = ("pdf", "png", "jpeg", "webp", "svg")
    # settings constraints
    PLOTLY_MIN_VER = ("5", "14", "0")
    PLOTLY_MAX_VER = ("6", "0", "0")
    KALEIDO_MIN_VER = ("0", "2", "1")
    KALEIDO_MAX_VER = ("0", "3", "0")

    def _check_plotly_kaleido_versions(self) -> None:
        self._plotly_installed = False
        self._plotly_ver_ok = False
        self._kaleido_installed = False
        self._kaleido_ver_ok = False

        try:
            import plotly
        except ImportError:
            pass
        else:
            self._plotly_installed = True
            plotly_ver = tuple([x for x in plotly.__version__.split(".")][:3])
            if plotly_ver >= self.PLOTLY_MIN_VER and plotly_ver < self.PLOTLY_MAX_VER:
                self._plotly_ver_ok = True

        try:
            import kaleido
        except ImportError:
            pass
        else:
            self._kaleido_installed = True
            kaleido_ver = tuple([x for x in kaleido.__version__.split(".")][:3])
            if kaleido_ver >= self.KALEIDO_MIN_VER and kaleido_ver < self.KALEIDO_MAX_VER:
                self._kaleido_ver_ok = True

    def _validate_figure_settings(self) -> None:
        self._check_plotly_kaleido_versions()

        figure_settings = self._settings.get("figure", {})

        if not isinstance(figure_settings, dict):
            raise SettingsError(SettingsError.FIGURE_KEY_TYPE)

        if figure_settings:
            # plotly library must be installed, and at least 5.14.0 version
            # to enable any figure instantiation/exportation
            if not self._plotly_installed:
                raise SettingsError(SettingsError.PLOTLY_NOT_INSTALLED)

            if not self._plotly_ver_ok:
                raise SettingsError(SettingsError.PLOTLY_VERSION)

            if "export" in figure_settings:
                export = figure_settings.get("export")

                if not isinstance(export, dict):
                    raise SettingsError(SettingsError.FIGURE_EXPORT_VALUE_TYPE)

                export_type = export.get("type")
                if export_type is None:
                    raise SettingsError(SettingsError.FIGURE_EXPORT_TYPE_MISSING)

                if export_type not in ("html", "image"):
                    raise SettingsError(SettingsError.FIGURE_EXPORT_TYPE_VALUE)

                export_path = export.get("path")
                if export_path is None:
                    raise SettingsError(SettingsError.FIGURE_EXPORT_PATH_MISSING)

                if not isinstance(export_path, str):
                    raise SettingsError(SettingsError.FIGURE_EXPORT_PATH_VALUE)

                export_path = Path(export_path)
                if not export_path.exists():
                    raise SettingsError(SettingsError.FIGURE_EXPORT_PATH_NOT_EXISTS)

                if not export_path.is_dir():
                    raise SettingsError(SettingsError.FIGURE_EXPORT_PATH_NOT_DIRECTORY)

                file_format = export.get("format")
                if file_format is None and export_type != "html":
                    raise SettingsError(SettingsError.FIGURE_EXPORT_FORMAT_MISSING)

                if export_type != "html" and not isinstance(file_format, str):
                    raise SettingsError(SettingsError.FIGURE_EXPORT_FORMAT_TYPE)

                accepted_formats = self.ACCEPTED_IMAGE_EXPORT_FORMATS
                if export_type == "image" and file_format not in accepted_formats:
                    raise SettingsError(SettingsError.FIGURE_EXPORT_FORMAT_VALUE)

                file_name = export.get("file_name", None)
                if file_name is None:
                    self._settings["figure"]["export"][
                        "file_name"
                    ] = self.FIGURE_DEFAULT_FILE_NAME
                else:
                    if not isinstance(file_name, str):
                        raise SettingsError(SettingsError.FIGURE_EXPORT_FILE_NAME_TYPE)

                    if not self.FIGURE_FILE_NAME_REGEX.match(file_name):
                        raise SettingsError(SettingsError.FIGURE_EXPORT_FILE_NAME_VALUE)

                if export_type == "image":
                    if not self._kaleido_installed:
                        raise SettingsError(SettingsError.FIGURE_EXPORT_KALEIDO_MISSING)

                    if not self._kaleido_ver_ok:
                        raise SettingsError(SettingsError.FIGURE_EXPORT_KALEIDO_VERSION)

                    export_width = export.get("width")
                    if export_width is not None:
                        if not isinstance(export_width, int) or export_width <= 0:
                            raise SettingsError(SettingsError.FIGURE_EXPORT_WIDTH_VALUE)
                    export_height = export.get("height")
                    if export_height is not None:
                        if not isinstance(export_height, int) or export_height <= 0:
                            raise SettingsError(SettingsError.FIGURE_EXPORT_HEIGHT_VALUE)

            show = figure_settings.get("show", False)
            if not isinstance(show, bool):
                raise SettingsError(SettingsError.FIGURE_SHOW_VALUE)

    def colorgen(self, index) -> str:
        """
        Method for returning a hexadecimal color for every item
        in the graph.
        """
        return constants.ITEMS_COLORS[index]

    def create_figure(self, show=False) -> None:
        """
        Method used for creating figures and showing/exporting them.

        **WARNING**
            plotly library at least 5.14.0 must be installed in environment,
            and for image exportation, at least kaleido 0.2.1.

            See :ref:`here<figures_guide>` for
            detailed explanation of the method.

        **OPERATION**
            Create's the solution's figure.

        **PARAMETERS**
            ``show``: if True, the created figure will be displayed in browser \
            after creation.
        """

        if not self.solution:
            hyperLogger.warning(FigureExportError.NO_SOLUTION_WARNING)
            return

        if not self._plotly_installed:
            raise SettingsError(SettingsError.PLOTLY_NOT_INSTALLED)

        elif not self._plotly_ver_ok:
            raise SettingsError(SettingsError.PLOTLY_VERSION)

        else:
            import plotly

            go = plotly.graph_objects

        figure_settings = self._settings.get("figure", {})
        export = figure_settings.get("export")
        show = figure_settings.get("show") or show

        if not show and export is None:
            hyperLogger.warning(FigureExportError.NO_FIGURE_OPERATION)
            return

        containers_ids = tuple(self._containers)

        for cont_id in containers_ids:
            W, L, H = self._containers[cont_id]["W"], self._containers[cont_id]["L"], self._containers._get_height(cont_id)
            box_traces = []
            for item_id in self.solution[cont_id]:
                item = self.solution[cont_id][item_id]
                Xo, Yo, Zo, w, l, h = item[0], item[1], item[2], item[3], item[4], item[5]

                vertices = [
                    (Xo, Yo, Zo),
                    (Xo + w, Yo, Zo),
                    (Xo + w, Yo + l, Zo),
                    (Xo, Yo + l, Zo),
                    (Xo, Yo, Zo + h),
                    (Xo + w, Yo, Zo + h),
                    (Xo + w, Yo + l, Zo + h),
                    (Xo, Yo + l, Zo + h)
                ]

                box_trace = go.Mesh3d(
                    x=[vertex[0] for vertex in vertices],
                    y=[vertex[1] for vertex in vertices],
                    z=[vertex[2] for vertex in vertices],
                    i=[0, 2, 4, 6, 0, 5, 3, 6, 0, 7, 1, 6],
                    j=[1, 3, 5, 7, 1, 4, 2, 7, 3, 4, 2, 5],
                    k=[2, 0, 6, 4, 5, 0, 6, 3, 7, 0, 6, 1],
                    name=item_id,
                    showlegend=True
                )
                box_traces.append(box_trace)

            fig = go.Figure(data=box_traces)

            fig.update_layout(
            title_text='3D Render of Boxes with Plotly',
            scene=dict(
                xaxis=dict(range=[0, W]),
                yaxis=dict(range=[0, L]),
                zaxis=dict(range=[0, H]),
                aspectratio=dict(x=W/(W+L), y=L/(W+L), z=H/(W+L)),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
)

            if export:
                try:
                    export_type = export.get("type", "html")
                    export_path = Path(export["path"])
                    file_name = export.get("file_name", "")

                    if export_type == "html":
                        fig.write_html(export_path / f"{file_name}__{cont_id}.html")

                    elif export_type == "image":
                        import plotly.io as pio

                        file_format = export["format"]
                        width = export.get("width") or 1700
                        height = export.get("height") or 1700
                        scale = 1
                        pio.kaleido.scope.default_width = width
                        pio.kaleido.scope.default_height = height
                        pio.kaleido.scope.default_scale = scale
                        fig.write_image(
                            export_path / f"{file_name}__{cont_id}.{file_format}"
                        )

                except Exception as e:
                    error_msg = FigureExportError.FIGURE_EXPORT.format(e)
                    raise FigureExportError(error_msg)
            if show:
                fig.show(config={"responsive": False})

    def validate_settings(self) -> None:
        self._validate_settings()
        self._validate_figure_settings()


class ItemsManipulationMixin:
    """
    Mixin providing ``items`` manipulation methods.
    """

    def orient_items(self, orientation: str or None = "short") -> None:
        """
        Method for orienting the ``items`` structure.

        **OPERATION**
            Orients each item in items set by rotating it
            (interchange w, l, h of item).

            See :ref:`here<orient_items>` for
            detailed explanation of the method.

        **PARAMETERS**
            ``orientation`` : "short"/"tall". If None provided, \
            orientation will be skipped.

        **WARNING**
            Changing the values of ``self.items`` causes
            resetting of the ``solution`` attribute.
        """
        if orientation is None:
            return

        if not self._rotation:
            hyperLogger.warning("can't rotate items. Rotation is disabled")
            return

        items = self._items

        if orientation not in ("short", "tall"):
            hyperLogger.warning(
                f"orientation parameter '{orientation}' not valid. Orientation skipped."
            )
            return

        for _id in items:
            if items[_id].get("horizontal_rotation_only", False):
                continue

            w, l, h = items[_id]["w"], items[_id]["l"], items[_id]["h"]

            if orientation == "short":
                if h > w:
                    h, w = w, h
                if h > l:
                    h, l = l, h
                items[_id]["w"], items[_id]["l"], items[_id]["h"] = w, l, h

            elif orientation == "tall":
                if h < l:
                    h, l = l, h
                if h < w:
                    h, w = w, h
                items[_id]["w"], items[_id]["l"], items[_id]["h"] = w, l, h

    def sort_items(self, sorting_by: tuple or None = ("volume", True)) -> None:
        """
        Method for ordering the ``items`` structure. See :ref:`here<sort_items>` for
        detailed explanation of the method.

        **OPERATION**
            Sorts the ``items``

            according to ``sorting_by`` parameter guidelines.

        **PARAMETERS**
            ``sorting_by`` : (sorting criterion, reverse). If None provided, sorting
            will be skipped.

        **WARNING**
            Changing the values of ``items`` causes resetting of the
            ``solution`` attribute.

        **RETURNS**
            ``None``
        """
        if sorting_by is None:
            return

        by, reverse = sorting_by

        items = self._items.deepcopy()

        if by == "volume":
            sorted_items = [[i["w"] * i["l"] * i["h"], _id] for _id, i in items.items()]
            sorted_items.sort(reverse=reverse)
        elif by == "surface_area":
            sorted_items = [[2 * i["w"] * i["l"] + 2 * i["w"] * i["h"] + 2 * i["l"] * i["h"], _id] for _id, i in items.items()]
            sorted_items.sort(reverse=reverse)
        elif by == "longest_side_ratio":
            sorted_items = [
                [max(i["w"], i["l"], i["h"]) / min(i["w"], i["l"], i["h"]), _id]
                for _id, i in items.items()
            ]
            sorted_items.sort(reverse=reverse)
        elif by == "largest_side_area":
            sorted_items = [
                [max(i["w"] * i["l"], i["w"] * i["h"], i["l"] * i["h"]), _id]
                for _id, i in items.items()
            ]
            sorted_items.sort(reverse=reverse)
        else:
            raise NotImplementedError

        self.items = {el[1]: items[el[1]] for el in sorted_items}


class DeepcopyMixin:
    """
    Mixin class providing copy/deepcopy utilities for
    certain attributes.
    """

    def _copy_objective_val_per_container(self, obj_val_per_container=None):
        if obj_val_per_container is None:
            obj_val_per_container = self.obj_val_per_container
        return {
            cont_id: obj_val_per_container[cont_id] for cont_id in obj_val_per_container
        }

    def _deepcopy_solution(self, solution=None):
        if solution is None:
            solution = self.solution
        return {
            cont_id: {
                item_id: [data for data in solution[cont_id].get(item_id, [])]
                for item_id in self._items
                if solution[cont_id].get(item_id) is not None
            }
            for cont_id in self._containers
        }


class LocalSearchMixin(AbstractLocalSearch, DeepcopyMixin):
    """
    Mixin implementing the Local Search.
    """

    def evaluate_node(self, sequence):
        self.solve(sequence=sequence, debug=False)

    def get_solution(self):
        return (
            self._deepcopy_solution(),
            self._copy_objective_val_per_container(),
        )

    def calculate_obj_value(self):
        """
        Calculates an objective value that prioritizes using the fewest containers.
        The score is primarily determined by the number of empty containers, with the
        average utilization of used containers acting as a tie-breaker.
        """
        if not self.solution:
            return 0

        total_containers_available = len(self._containers)
        used_containers_count = 0
        sum_of_utilizations = 0.0
        total_penalty = 0.0

        weights = {cont_id: 1 if i != total_containers_available-1 else 0.7 for i, cont_id in enumerate(self._containers)}

        # Iterate through each container that has a solution
        for cont_id in self.solution:
            container = self._containers[cont_id]
            solution_items = self.solution[cont_id]

            # Only consider containers that have items in them
            if not solution_items:
                continue
            
            used_containers_count += 1

            H = self._get_initial_container_height(container)
            total_container_volume = container["W"] * container["L"] * H
            
            total_items_volume = 0
            weighted_z_sum = 0

            for item_id, item_data in solution_items.items():
                # item_data from solution is [Xo, Yo, Zo, w, l, h]
                item_volume = item_data[3] * item_data[4] * item_data[5]
                total_items_volume += item_volume
                # Calculate center of mass using the item's center point
                item_center_z = item_data[2] + (item_data[5] / 2)
                weighted_z_sum += item_volume * item_center_z

            # Calculate utilization and penalty for this specific container
            utilization = total_items_volume / total_container_volume
            sum_of_utilizations += utilization * weights[cont_id]

            center_of_mass_z = weighted_z_sum / total_items_volume
            normalized_penalty = center_of_mass_z / H
            penalty_factor = 0.01  # Small factor to act as a micro-tie-breaker
            total_penalty += penalty_factor * normalized_penalty

        # If no items were packed at all
        if used_containers_count == 0:
            return 0

        # Heuristic Calculation
        # Major component: Number of empty containers. Using a large constant (1.0) for each.
        # Minor component: Average utilization of the containers that were actually used.
        empty_containers_score = total_containers_available - used_containers_count
        average_utilization = sum_of_utilizations / used_containers_count
        
        # The final score prioritizes leaving bins empty, then packing the used bins well.
        # The Z-mass penalty acts as a final small tie-breaker.
        return empty_containers_score + sum_of_utilizations - total_penalty

    def get_init_solution(self):
        self.solve(debug=False)
        # deepcopying solution
        best_solution = self._deepcopy_solution()
        best_obj_val_per_container = self._copy_objective_val_per_container()
        return best_solution, best_obj_val_per_container

    def extra_node_operations(self, **kwargs):
        if self._strip_pack:
            # new height is used for the container
            # for neighbors of new node
            self._containers._set_height()
            self._heights_history.append(self._container_height)

    def compare_node(self, new_obj_value, best_obj_value):
        """
        Used in local_search.
        Compares new solution value to best for accepting new node. It's the
        mechanism for propagating towards new accepted better solutions/nodes.

        In bin-packing mode, a simple comparison using solution_operator is made.

        In strip-packing mode, extra conditions will be tested:

            - If ``self._container_min_height`` is ``None``:
                The total of items  must be in solution. \
                If not, solution is rejected.

            - If ``self._container_min_height`` is not ``None``:
                Number of items in solution doesn't affect \
                solution choice.
        """
        better_solution = new_obj_value > best_obj_value

        if not self._strip_pack:
            return better_solution

        if self._container_min_height is None:
            extra_cond = len(self.solution[self.STRIP_PACK_CONT_ID]) == len(self._items)
        else:
            extra_cond = True

        return extra_cond and better_solution

    def local_search(
        self, *, throttle: bool = True, _hypersearch: bool = False, debug: bool = False
    ) -> None:
        """
        Method for deploying a hill-climbing local search operation, using the
        default potential points strategy. Solves against the ``self.items`` and
        the ``self.containers`` attributes.

        **OPERATION**
            Updates ``self.solution`` with the best solution found.

            Updates ``self.obj_val_per_container`` with the best values found.

        **PARAMETERS**
            ``throttle`` affects the total neighbors parsing before accepting that
            no better node can be found. Aims at containing the total execution time
            in big instances of the problem. Corresponds to ~ 72 items instance
            (max 2500 neighbors).

            ``_hypersearch``: Either standalone (False), or part of a
            superset search (used by hypersearch).

            ``debug``: for developing debugging.

        **RETURNS**
            ``None``
        """

        if not _hypersearch:
            start_time = time.time()
        else:
            start_time = self.start_time
            hyperLogger.debug(
                "\t\tCURRENT POTENTIAL POINTS STRATEGY:"
                f" {self._potential_points_strategy}"
            )

        if self._strip_pack:
            self._heights_history = [self._container_height]

        # after local search has ended, restore optimum values
        # retain_solution = (solution, obj_val_per_container)
        retained_solution = super().local_search(
            list(self._items),
            throttle,
            start_time,
            self._max_time_in_seconds,
            debug=debug,
        )
        self.solution, self.obj_val_per_container = retained_solution
