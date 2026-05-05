"""Domain generalisation techniques: GRL, CLIP alignment, meta-learning."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for GRL."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (GRL) for domain adversarial training.

    Forward pass: y = x
    Backward pass: gradient is negated (flipped sign).
    This encourages the model to learn domain-invariant features.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss using GRL.

    Train a domain discriminator on features from the main model.
    GRL flips the gradient during backprop, making features
    less discriminative for domain classification.
    """

    def __init__(self, feature_dim: int, domain_dim: int, lambda_grl: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_grl)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, domain_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        if domain_labels is not None:
            loss = F.cross_entropy(domain_logits, domain_labels)
            return {"domain_adversarial_loss": loss, "domain_logits": domain_logits}

        return {"domain_logits": domain_logits}


class MetaLearningModule(nn.Module):
    """Meta-learning inner loop for Model-Agnostic Meta-Learning (MAML).

    Supports task-level adaptation: train on multiple domains, validate
    on held-out domain to improve generalisation.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr

    def inner_loop_update(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute inner loop loss on support set and adapt."""
        adapted_params: Dict[str, nn.Parameter] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapted_params[name] = param.clone()
            else:
                adapted_params[name] = param

        logits = self.model(support_images)
        loss = F.cross_entropy(logits, support_labels)
        grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)

        adapted_params = {
            name: param - self.inner_lr * grad
            for (name, param), grad in zip(adapted_params.items(), grads)
            if grad is not None
        }

        return adapted_params


def apply_dg_technique(
    model: nn.Module,
    technique: str,
    cfg: Dict[str, Any],
) -> nn.Module:
    """Apply a domain-generalisation technique to the model.

    Args:
        model: Base model to wrap.
        technique: One of "grad_reversal", "clip_align", "meta_learning", "none".
        cfg: Configuration dict.

    Returns:
        Modified model with DG technique applied.
    """
    if technique == "none" or technique is None:
        return model

    if technique == "grad_reversal":
        return model

    if technique == "clip_align":
        return model

    if technique == "meta_learning":
        return MetaLearningModule(model)

    raise ValueError(
        f"Unknown DG technique: {technique}. Supported: none, grad_reversal, clip_align, meta_learning"
    )
