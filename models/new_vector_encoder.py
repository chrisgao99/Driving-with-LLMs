from math import sqrt

import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import Perceiver
# NOTE: Ensure the path to vector_utils is correct for your project structure
from utils.new_vector_utils import VectorObservation, VectorObservationConfig


class VectorEncoderConfig:
    model_dim: int = 256
    num_latents: int = 32
    num_blocks: int = 7
    num_heads: int = 8


class VectorEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: VectorEncoderConfig,
        observation_config: VectorObservationConfig,
        num_queries: int,
    ):
        super().__init__()

        model_dim = encoder_config.model_dim
        
        ## NOTE: The input dimensions for the MLPs have been updated
        ## to match the new data format from VectorObservation.
        self.ego_vehicle_encoder = MLP(
            VectorObservation.EGO_DIM, [model_dim], model_dim
        )
        self.vehicle_encoder = MLP(
            VectorObservation.VEHICLE_DIM, [model_dim], model_dim
        )
        self.pedestrian_encoder = MLP(
            VectorObservation.PEDESTRIAN_DIM, [model_dim], model_dim
        )
        # The input dimension is updated to ROAD_DIM (46)
        self.road_encoder = MLP(VectorObservation.ROAD_DIM, [model_dim], model_dim)
        
        # The embedding size is updated to the number of road points (60)
        # self.road_embedding = nn.Parameter(
        #     torch.randn((observation_config.num_road_points, model_dim))
        #     / sqrt(model_dim)
        # )

        self.perceiver = Perceiver(
            model_dim=model_dim,
            context_dim=model_dim,
            num_latents=encoder_config.num_latents,
            num_blocks=encoder_config.num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_queries,
        )

        self.out_features = model_dim

    def forward(self, obs: VectorObservation):
        ## NOTE: The forward pass now uses `road_descriptors` instead of `route_descriptors`.
        batch = obs.road_descriptors.shape[0]
        device = obs.road_descriptors.device

        # Encode road, vehicle, and pedestrian information
        road_token = self.road_encoder(obs.road_descriptors)
        vehicle_token = self.vehicle_encoder(obs.vehicle_descriptors)
        pedestrian_token = self.pedestrian_encoder(obs.pedestrian_descriptors)

        # Concatenate all context tokens
        context = torch.cat((road_token, pedestrian_token, vehicle_token), dim=-2)
        
        # Create a context mask. It's assumed that an active slot has a non-zero
        # value in its first feature (index 0).
        context_mask = torch.cat(
            (
                # Road descriptors are assumed to be always active
                torch.ones(
                    (batch, road_token.shape[1]), device=device, dtype=torch.bool
                ),
                # Mask for active pedestrians
                obs.pedestrian_descriptors[:, :, 0] != 0,
                # Mask for active vehicles
                obs.vehicle_descriptors[:, :, 0] != 0,
            ),
            dim=1,
        )

        # Encode the ego vehicle state
        ego_vehicle_state = obs.ego_vehicle_descriptor
        ego_vehicle_feat = self.ego_vehicle_encoder(ego_vehicle_state)

        # Process through the Perceiver model
        feat, _ = self.perceiver(ego_vehicle_feat, context, context_mask=context_mask)
        feat = feat.view(
            batch,
            self.perceiver.num_queries,
            feat.shape[-1],
        )

        return feat