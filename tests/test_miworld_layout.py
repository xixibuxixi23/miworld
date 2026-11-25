import json
from pathlib import Path

import pytest
import torch
from torch import nn

from miworld.models.m3 import M3, M3Policy, M3PolicyArgs
from miworld.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs


class DummyRenderer(nn.Module):
    def forward(self, value):
        return value + 1


def test_m3_renderer_only_facade():
    model = M3(DummyRenderer())
    assert not model.policy_enabled
    assert torch.equal(model.forward_renderer(torch.tensor(2)), torch.tensor(3))
    with pytest.raises(RuntimeError, match="without a policy checkpoint"):
        model.forward_policy()


def test_m3_policy_names_are_checkpoint_compatible_aliases():
    assert M3Policy is InsertedInhabitantPolicy
    assert M3PolicyArgs is InsertedPolicyArgs


def test_latest_model_registry_has_only_three_public_models():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs/models/latest.json").read_text())
    assert set(registry["models"]) == {"m1", "m2", "m3"}
    assert registry["models"]["m3"]["renderer_checkpoint"]
    assert registry["models"]["m3"]["policy_checkpoint"]
