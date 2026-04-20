from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class ClientSplit:
    patient_id: str
    frame: pd.DataFrame


def split_by_patient(frame: pd.DataFrame) -> list[ClientSplit]:
    splits: list[ClientSplit] = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        splits.append(ClientSplit(patient_id=str(patient_id), frame=group.copy()))
    return splits


def weighted_average_state_dicts(state_dicts: list[dict], weights: list[float]) -> dict:
    if not state_dicts:
        raise ValueError("No client state dicts provided")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    keys = state_dicts[0].keys()
    averaged = {}
    for key in keys:
        template = state_dicts[0][key]
        if hasattr(template, "is_floating_point") and template.is_floating_point():
            value = None
            for state, weight in zip(state_dicts, weights):
                tensor = state[key]
                contribution = tensor.detach().clone() * (weight / total)
                value = contribution if value is None else value + contribution
            averaged[key] = value
        else:
            averaged[key] = template.detach().clone()
    return averaged


def fit_federated_rounds(
    client_splits: Iterable[ClientSplit],
    model_factory: Callable[[], object],
    local_train_fn: Callable[[object, pd.DataFrame], dict],
    rounds: int = 1,
) -> object:
    """Generic FedAvg loop for PyTorch models."""
    clients = list(client_splits)
    if not clients:
        raise ValueError("No clients available for federated training")
    global_model = model_factory()
    for _ in range(rounds):
        local_states = []
        weights = []
        for client in clients:
            local_model = model_factory()
            local_model.load_state_dict(global_model.state_dict())
            state = local_train_fn(local_model, client.frame)
            local_states.append(state)
            weights.append(float(len(client.frame)))
        averaged = weighted_average_state_dicts(local_states, weights)
        global_model.load_state_dict(averaged)
    return global_model
