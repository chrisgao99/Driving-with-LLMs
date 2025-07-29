import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

# --- Randomization utils ---
class Randomizable:
    ENUM: Any = None
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {}

    @classmethod
    def randomize(cls, vector: np.ndarray):
        for field_name, (field_type, field_range) in cls.FIELD_TYPES_RANGES.items():
            idx = getattr(cls.ENUM, field_name)
            vector[idx] = random_value((field_type, field_range))


def random_value(
    field_type_range: Tuple[type, Tuple[float, float]],
    prob: float = 0.5,
) -> Union[int, float]:
    field_type, field_range = field_type_range
    if field_type == bool:
        return 1 if random.random() < prob else 0
    if field_type == float:
        return random.uniform(*field_range)
    if field_type == int:
        return random.randint(*field_range)
    raise ValueError(f"Unsupported field type: {field_type}")


class VehicleFieldRandom(Randomizable):
    ENUM = VehicleField
    FIELD_TYPES_RANGES: Dict[str, Any] = {
        "SPEED": (float, (0.0, 40.0)),  # 0 to ~90 mph
        "X": (float, (-100.0, 100.0)), # Relative position in meters
        "Y": (float, (-100.0, 100.0)),
        "DX": (float, (-1.0, 1.0)),     # Normalized direction vector
        "DY": (float, (-1.0, 1.0)),
        "HEADING": (float, (-math.pi, math.pi)), # Radians
    }


class PedestrianFieldRandom(Randomizable):
    ENUM = PedestrianField
    FIELD_TYPES_RANGES: Dict[str, Any] = {
        "SPEED": (float, (0.0, 3.0)),   # 0 to ~7 mph (running)
        "X": (float, (-50.0, 50.0)),    # Pedestrians are usually closer
        "Y": (float, (-50.0, 50.0)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
        "HEADING": (float, (-math.pi, math.pi)),
    }


class EgoFieldRandom(Randomizable):
    ENUM = EgoField
    FIELD_TYPES_RANGES: Dict[str, Any] = {
        "ACCEL": (float, (-5.0, 3.0)), # m/s^2 (hard brake to strong accel)
        "SPEED": (float, (0.0, 40.0)),
        "X": (float, (-50.0, 50.0)),
        "Y": (float, (-50.0, 50.0)),      
        "HEADING": (float, (-math.pi, math.pi)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
    }


class RoadFieldRandom:
    @staticmethod
    def randomize(vector: np.ndarray):
        # --- Randomize Polyline Points and Validity ---
        num_valid_points = random.randint(2, 10) # Road features have at least 2 points
        last_x, last_y = 0, 0
        for i in range(10):
            if i < num_valid_points:
                # Point is valid
                vector[getattr(RoadField, f"INVALID_{i+1}")] = 0
                # Generate a new point relative to the last one to create a smooth line
                last_x += random.uniform(-10, 10)
                last_y += random.uniform(-10, 10)
                vector[getattr(RoadField, f"X{i+1}")] = last_x
                vector[getattr(RoadField, f"Y{i+1}")] = last_y
            else:
                # Point is invalid
                vector[getattr(RoadField, f"INVALID_{i+1}")] = 1
                # Zero out the coordinates for invalid points
                vector[getattr(RoadField, f"X{i+1}")] = 0
                vector[getattr(RoadField, f"Y{i+1}")] = 0

        # --- Randomize Type (One-Hot Encoding) ---
        type_fields = [field for field in RoadField if field.name.startswith("TYPE_")]
        # Reset all type fields to 0
        for field in type_fields:
            vector[field.value] = 0
        # Pick one random type and set it to 1
        random_type = random.choice(type_fields)
        vector[random_type.value] = 1


@dataclass
class VectorObservation:
    ROAD_DIM = 46
    VEHICLE_DIM = 6
    PEDESTRIAN_DIM = 6
    EGO_DIM = 7

    road_descriptors: torch.FloatTensor
    vehicle_descriptors: torch.FloatTensor
    pedestrian_descriptors: torch.FloatTensor
    ego_vehicle_descriptor: torch.FloatTensor
    liable_vehicles: Optional[torch.FloatTensor] = None


class VectorObservationConfig:
    num_road_points: int = 60
    num_vehicle_slots: int = 30
    num_pedestrian_slots: int = 20

    radius_m: float = 100.0
    pedestrian_radius_m: float = 50.0
    pedestrian_angle_threshold_rad: float = math.pi / 2
    route_spacing_m: float = 2.0
    num_max_static_vehicles: int = 10
    line_of_sight: bool = False
