import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Setting up scale factors
METRES_M_SCALE = 10.0
MS_TO_MPH = 2.23694
VELOCITY_MS_SCALE = 5.0


# Enumerating fields for the vector representations of the different objects
# Unspecified fields are not involved in the structured language generation
class VehicleField(IntEnum):
    SPEED = 0
    X = 1
    Y = 2
    DX = 3
    DY = 4
    HEADING = 5


class PedestrianField(IntEnum):
    SPEED = 0
    X = 1
    Y = 2
    DX = 3
    DY = 4
    HEADING = 5


class RoadField(IntEnum):
    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3  
    X3 = 4
    Y3 = 5
    X4 = 6
    Y4 = 7
    X5 = 8
    Y5 = 9
    X6 = 10
    Y6 = 11
    X7 = 12
    Y7 = 13
    X8 = 14
    Y8 = 15
    X9 = 16
    Y9 = 17
    X10 = 18
    Y10 = 19
    INVALID_1 = 20
    INVALID_2 = 21
    INVALID_3 = 22
    INVALID_4 = 23
    INVALID_5 = 24
    INVALID_6 = 25
    INVALID_7 = 26
    INVALID_8 = 27
    INVALID_9 = 28
    INVALID_10 = 29
    TYPE_LANE_CENTER_FREEWAY = 30
    TYPE_LANE_CENTER_SURFACE_STREET = 31
    TYPE_LANE_CENTER_BIKE_LANE = 32
    TYPE_ROAD_LINE_BROKEN_SINGLE_WHITE = 33
    TYPE_ROAD_LINE_SOLID_SINGLE_WHITE = 34
    TYPE_ROAD_LINE_SOLID_DOUBLE_WHITE = 35
    TYPE_ROAD_LINE_BROKEN_SINGLE_YELLOW = 36
    TYPE_ROAD_LINE_BROKEN_DOUBLE_YELLOW = 37
    TYPE_ROAD_LINE_SOLID_SINGLE_YELLOW = 38
    TYPE_ROAD_LINE_SOLID_DOUBLE_YELLOW = 39
    TYPE_ROAD_LINE_PASSING_DOUBLE_YELLOW = 40
    TYPE_ROAD_EDGE_BOUNDARY = 41
    TYPE_ROAD_EDGE_MEDIAN = 42
    TYPE_STOP_SIGN = 43
    TYPE_CROSSWALK = 44
    TYPE_SPEED_BUMP = 45


class EgoField(IntEnum):
    ACCEL = 0
    SPEED = 1
    X = 2
    Y = 3
    HEADING = 4
    DX = 5
    DY = 6


class LiableVechicleField(IntEnum):
    VEHICLE_0 = 0
    VEHICLE_1 = 1
    VEHICLE_2 = 2
    VEHICLE_3 = 3
    VEHICLE_4 = 4
    VEHICLE_5 = 5
    VEHICLE_6 = 6
    VEHICLE_7 = 7
    VEHICLE_8 = 8
    VEHICLE_9 = 9
    VEHICLE_13 = 13
    VEHICLE_14 = 14
    VEHICLE_15 = 15
    VEHICLE_16 = 16
    VEHICLE_17 = 17
    VEHICLE_18 = 18
    VEHICLE_19 = 19
    VEHICLE_20 = 20
    VEHICLE_21 = 21
    VEHICLE_22 = 22
    VEHICLE_23 = 23
    VEHICLE_24 = 24
    VEHICLE_25 = 25
    VEHICLE_26 = 26
    VEHICLE_27 = 27
    VEHICLE_28 = 28
    VEHICLE_29 = 29
