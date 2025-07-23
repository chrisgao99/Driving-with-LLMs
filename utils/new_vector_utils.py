import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union
from data_conversion.vector_utils_waymo_custom import VehicleField, PedestrianField, RoadField, EgoField, METRES_M_SCALE, VELOCITY_MS_SCALE, MS_TO_MPH
import numpy as np
import torch


# --- Utility functions ---
## NOTE: These functions are updated to use the new IntEnum field definitions.

def xy_from_vehicle_desc(vehicle_array):
    x = vehicle_array[:, VehicleField.X]
    y = vehicle_array[:, VehicleField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def traveling_angle_deg_from_vehicle_desc(vehicle_array):
    dx = vehicle_array[:, VehicleField.DX]
    dy = vehicle_array[:, VehicleField.DY]
    return direction_to_angle_deg(dx, dy)


def speed_mph_from_vehicle_desc(vehicle_array):
    return vehicle_array[:, VehicleField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH


def xy_from_pedestrian_desc(pedestrian_array):
    x = pedestrian_array[:, PedestrianField.X]
    y = pedestrian_array[:, PedestrianField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def traveling_angle_deg_from_pedestrian_desc(pedestrian_array):
    dx = pedestrian_array[:, PedestrianField.DX]
    dy = pedestrian_array[:, PedestrianField.DY]
    return direction_to_angle_deg(dx, dy)


def start_xy_from_road_desc(road_array):
    """Extracts the first (X, Y) coordinate pair from each road element."""
    x = road_array[:, RoadField.X1]
    y = road_array[:, RoadField.Y1]
    return torch.vstack((x, y)).T * METRES_M_SCALE


def flags_in_fov(xy_coords, fov_degrees=60, max_distance=40):
    """Filters coordinates to find those within a forward-facing FOV."""
    distances, angular = angles_deg_and_distances(xy_coords)
    return (
        (xy_coords[:, 0] > 0)
        & (torch.abs(angular) < fov_degrees / 2)
        & (distances <= max_distance)
    )


def angles_deg_and_distances(xy_coords):
    distances = torch.linalg.norm(xy_coords, axis=1)
    angular = direction_to_angle_deg(xy_coords[:, 0], xy_coords[:, 1])
    return distances, angular


def direction_to_angle_deg(dirx, diry):
    return torch.atan2(-diry, dirx) * 180.0 / np.pi


def vehicle_filter_flags(vehicle_descriptors):
    """
    Generates boolean flags for active vehicles within the field of view.
    NOTE: Assumes a vehicle is 'active' if its speed is greater than 0,
    as there is no longer an explicit ACTIVE flag.
    """
    active_flags = vehicle_descriptors[:, VehicleField.SPEED] > 0
    fov_flags = flags_in_fov(xy_from_vehicle_desc(vehicle_descriptors), max_distance=40)
    return active_flags & fov_flags


def pedestrian_filter_flags(pedestrian_descriptors):
    """
    Generates boolean flags for active pedestrians within the field of view.
    NOTE: Assumes a pedestrian is 'active' if their speed is greater than 0.
    """
    active_flags = pedestrian_descriptors[:, PedestrianField.SPEED] > 0
    fov_flags = flags_in_fov(
        xy_from_pedestrian_desc(pedestrian_descriptors), max_distance=30
    )
    return active_flags & fov_flags


@dataclass
class VectorObservation:
    """
    Vectorized representation for the new data format.
    It stores information about the environment in float torch tensors, coding flags and properties
    about the road, nearby vehicles, pedestrians etc.
    """
    ROAD_DIM = 46
    VEHICLE_DIM = 6
    PEDESTRIAN_DIM = 6
    EGO_DIM = 7

    # A 2d array describing the road and environment.
    road_descriptors: torch.FloatTensor

    # A 2d array describing nearby vehicles.
    vehicle_descriptors: torch.FloatTensor

    # A 2d array describing pedestrians.
    pedestrian_descriptors: torch.FloatTensor

    # A 1D array describing the ego vehicle's state.
    ego_vehicle_descriptor: torch.FloatTensor

    liable_vehicles: Optional[torch.FloatTensor] = None


class VectorObservationConfig:
    """Configuration for the observation vector shapes."""
    num_road_points: int = 60
    num_vehicle_slots: int = 30
    num_pedestrian_slots: int = 20

    radius_m: float = 100.0
    pedestrian_radius_m: float = 50.0
    pedestrian_angle_threshold_rad: float = math.pi / 2
    route_spacing_m: float = 2.0
    num_max_static_vehicles: int = 10
    line_of_sight: bool = False

# --- Randomization utils for generating synthetic data could be defined here ---
# --- They would need to be updated to match the new IntEnum classes and data structures. ---

