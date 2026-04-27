"""Sim4EndoR-aligned failure penalty: R_fail when guidewire exits vessel
or simulation error occurs."""
from .reward import Reward
from ..intervention import Intervention
from ..intervention.vesseltree.vesseltree import at_tree_end
from ..util.coordtransform import tracking3d_to_vessel_cs


class FailurePenalty(Reward):
    """Returns `factor` (e.g. -100) on vessel exit or sim error."""

    def __init__(self, intervention: Intervention, factor: float = -100.0) -> None:
        self.intervention = intervention
        self.factor = factor

    def step(self) -> None:
        failed = self.intervention.simulation.simulation_error

        if not failed:
            tip = self.intervention.fluoroscopy.tracking3d[0]
            tip_vessel = tracking3d_to_vessel_cs(
                tip,
                self.intervention.fluoroscopy.image_rot_zx,
                self.intervention.fluoroscopy.image_center,
            )
            failed = at_tree_end(tip_vessel, self.intervention.vessel_tree)

        self.reward = self.factor if failed else 0.0

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
