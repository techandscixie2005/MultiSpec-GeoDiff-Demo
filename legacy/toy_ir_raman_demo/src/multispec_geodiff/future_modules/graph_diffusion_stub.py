from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GraphEditOperation:
    name: str
    description: str


class SizeAdaptiveGraphDiffusionStub:
    """Interface placeholder for a future size-adaptive graph diffusion generator.

    A production version would condition on fused spectral evidence and learn graph edit
    trajectories with operations such as insert_node, delete_node, change_atom_type,
    add_bond, delete_bond, and change_bond_type. This demo intentionally does not fake
    training or sampling quality; the retrieval module is the current MVP substitute.
    """

    SUPPORTED_OPERATIONS = [
        GraphEditOperation("insert_node", "Insert a new atom node conditioned on spectra."),
        GraphEditOperation("delete_node", "Remove an atom node during size-adaptive refinement."),
        GraphEditOperation("change_atom_type", "Change atom identity while preserving graph context."),
        GraphEditOperation("add_bond", "Create a bond between compatible nodes."),
        GraphEditOperation("delete_bond", "Remove a bond when spectral evidence rejects it."),
        GraphEditOperation("change_bond_type", "Modify bond order during denoising."),
    ]

    def plan_sampling(self, spectral_condition: dict[str, Any], max_steps: int = 32) -> dict[str, Any]:
        return {
            "status": "stub",
            "message": "Graph diffusion is roadmap-only in this demo.",
            "max_steps": max_steps,
            "supported_operations": [op.name for op in self.SUPPORTED_OPERATIONS],
            "spectral_condition_keys": sorted(spectral_condition.keys()),
        }
