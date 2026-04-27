"""
SynthEndoSim — Guidewire device implementations.

Provides:
  JShapedGuidewire  — Standard J-tip 0.035" guidewire (most common CAS)
  StraightGuidewire — Straight stiff guidewire
  HydrophilicGuidewire — Slippery hydrophilic (lower friction, longer tip)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from .base import Device, SofaDeviceParams


@dataclass
class JShapedGuidewire(Device):
    """
    Standard J-tipped 0.035" guidewire.
    Default workhorse for CAS/EVAR navigation.

    Parameters
    ----------
    total_length    : float   total wire length (mm), default 450
    tip_length      : float   shaped J-tip section length (mm), default 40
    tip_angle       : float   J-bend half-angle (rad), default 0.21
    diameter        : float   wire outer diameter (mm), default 0.89 (0.035")
    young_modulus   : float   shaft Young's modulus (Pa)
    """
    total_length: float = 450.0
    tip_length: float = 40.0
    tip_angle: float = 0.21
    diameter: float = 0.89
    young_modulus: float = 170_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "j_guidewire"

    @property
    def sofa_params(self) -> SofaDeviceParams:
        straight = self.total_length - self.tip_length
        spire_d = 2.0 * self.tip_length / math.pi * self.tip_angle
        r = self.diameter / 2.0
        return SofaDeviceParams(
            is_a_procedural_shape=True,
            straight_length=straight,
            length=self.total_length,
            spire_diameter=spire_d,
            spire_height=0.0,
            radius_extremity=r,
            young_modulus_extremity=self.young_modulus * 0.5,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, straight, self.total_length],
            density_of_beams=[40, 10],
            num_edges_collis=[20, 10],
            num_edges=50,
        )

    @property
    def color(self):
        return (0.1, 0.1, 0.8, 1.0)


@dataclass
class StraightGuidewire(Device):
    """Straight stiff guidewire — used for initial access or stiff exchanges."""
    total_length: float = 450.0
    diameter: float = 0.89
    young_modulus: float = 200_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "straight_guidewire"

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
            young_modulus_extremity=self.young_modulus,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, self.total_length],
            density_of_beams=[50],
            num_edges_collis=[30],
            num_edges=50,
        )

    @property
    def color(self):
        return (0.8, 0.8, 0.1, 1.0)


@dataclass
class HydrophilicGuidewire(Device):
    """
    Hydrophilic-coated guidewire with longer, more flexible tip.
    Use lower friction coefficient in SofaBackend (friction=0.05).
    """
    total_length: float = 450.0
    tip_length: float = 80.0
    tip_angle: float = 0.15
    diameter: float = 0.89
    young_modulus: float = 120_000.0
    poisson_ratio: float = 0.49

    @property
    def name(self) -> str:
        return "hydrophilic_guidewire"

    @property
    def sofa_params(self) -> SofaDeviceParams:
        straight = self.total_length - self.tip_length
        spire_d = 2.0 * self.tip_length / math.pi * self.tip_angle
        r = self.diameter / 2.0
        return SofaDeviceParams(
            is_a_procedural_shape=True,
            straight_length=straight,
            length=self.total_length,
            spire_diameter=spire_d,
            spire_height=0.0,
            radius_extremity=r,
            young_modulus_extremity=self.young_modulus * 0.3,
            mass_density_extremity=0.000155,
            radius=r,
            young_modulus=self.young_modulus,
            mass_density=0.000155,
            poisson_ratio=self.poisson_ratio,
            key_points=[0.0, straight, self.total_length],
            density_of_beams=[35, 15],
            num_edges_collis=[18, 12],
            num_edges=50,
        )

    @property
    def color(self):
        return (0.1, 0.7, 0.3, 1.0)
