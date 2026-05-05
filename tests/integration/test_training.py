"""Integration tests for training pipelines."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.training.single_source import run_train_epoch
from src.training.multi_source import MultiSourceDataLoader
from src.training.domain_gen import (
    GradientReversalLayer,
    DomainAdversarialLoss,
    apply_dg_technique,
    MetaLearningModule,
)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"img": torch.randn(3, 224, 224), "labels": torch.randint(0, 10, (3,))}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class TestMultiSourceDataLoader:
    def test_combined_length(self):
        dl1 = torch.utils.data.DataLoader(DummyDataset(10), batch_size=2)
        dl2 = torch.utils.data.DataLoader(DummyDataset(5), batch_size=2)
        multi = MultiSourceDataLoader([dl1, dl2], [1.0, 1.0])
        assert len(multi) == 15

    def test_yields_batches(self):
        dl1 = torch.utils.data.DataLoader(DummyDataset(10), batch_size=2)
        dl2 = torch.utils.data.DataLoader(DummyDataset(5), batch_size=2)
        multi = MultiSourceDataLoader([dl1, dl2], [1.0, 1.0])

        batches = list(iter(multi))
        assert len(batches) == 15


class TestGradientReversalLayer:
    def test_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 10)
        assert torch.allclose(grl(x), x)


class TestApplyDGTechnique:
    def test_none_returns_model(self):
        model = DummyModel()
        result = apply_dg_technique(model, "none", {})
        assert result is model

    def test_meta_learning_wraps_model(self):
        model = DummyModel()
        result = apply_dg_technique(model, "meta_learning", {})
        assert isinstance(result, MetaLearningModule)

    def test_unknown_raises(self):
        model = DummyModel()
        with pytest.raises(ValueError, match="Unknown DG technique"):
            apply_dg_technique(model, "unknown", {})
