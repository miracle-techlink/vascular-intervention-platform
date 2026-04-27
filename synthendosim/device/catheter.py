"""
SynthEndoSim — Catheter device implementations.

Provides:
  SimmonsCatheter    — Simmons / Sidewinder shape (complex aortic arch)
  JudkinsCatheter    — Judkins Right/Left (coronary access)
  PigtailCatheter    — Pigtail (aortography)
  SheathCatheter     — Introducer sheath (outer device for co-axial systems)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from .base import Device, SofaDeviceParams


@dataclass
class SimmonsCatheter(Device):
    """
    Simmons (Sidewinder) catheter — self-forming loop for
    retrograde navigation of sharply angled aortic arch vessels.
    """
    total_length: float = 1000.0
    tip_length: float = 60.0
    diameter: float = 1.67      # 5F = 1.67mm outer diameter
    young_modulus: float = 80_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "simmons_catheter"

    @property
    def sofa_params(self) -> SofaDeviceParams:
        straight = self.total_length - self.tip_length
        r = self.diameter / 2.0
        return SofaDeviceParams(
            is_a_procedural_shape=True,
            straight_length=straight,
            length=self.total_length,
            spire_diameter=self.tip_length * 0.6,
            spire_height=self.tip_length * 0.3,
            radius_extremity=r * 0.9,
            young_modulus_extremity=self.young_modulus * 0.4,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, straight, self.total_length],
            density_of_beams=[30, 20],
            num_edges_collis=[15, 12],
            num_edges=50,
        )

    @property
    def color(self):
        return (0.8, 0.2, 0.1, 1.0)


@dataclass
class PigtailCatheter(Device):
    """Pigtail catheter — used for aortography and haemodynamic assessment."""
    total_length: float = 1000.0
    tip_coil_diameter: float = 12.0
    diameter: float = 1.33      # 4F
    young_modulus: float = 75_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "pigtail_catheter"

    @property
    def sofa_params(self) -> SofaDeviceParams:
        straight = self.total_length - 30.0
        r = self.diameter / 2.0
        return SofaDeviceParams(
            is_a_procedural_shape=True,
            straight_length=straight,
            length=self.total_length,
            spire_diameter=self.tip_coil_diameter,
            spire_height=0.0,
            radius_extremity=r * 0.85,
            young_modulus_extremity=self.young_modulus * 0.3,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, straight, self.total_length],
            density_of_beams=[35, 15],
            num_edges_collis=[18, 10],
            num_edges=50,
        )

    @property
    def color(self):
        return (0.7, 0.1, 0.7, 1.0)


@dataclass
class SheathCatheter(Device):
    """
    Introducer sheath — outer tube in co-axial system.
    Typically paired with a guidewire or inner catheter.
    """
    total_length: float = 800.0
    diameter: float = 2.67      # 8F outer
    young_modulus: float = 120_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "sheath"

    @property
    def sofa_params(self) -> SofaDeviceParams:
        r = self.diameter / 2.0
        return SofaDeviceParams(
            is_a_procedural_shape=True,
            straight_length=self.total_length,
            length=self.total_length,
            spire_diameter=4.0,
            spire_height=0.0,
            radius_extremity=r,
            young_modulus_extremity=self.young_modulus * 0.8,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, self.total_length],
            density_of_beams=[40],
            num_edges_collis=[25],
            num_edges=40,
        )

    @property
    def color(self):
        return (0.6, 0.6, 0.6, 0.7)
