"""
SynthEndoSim — SOFA + BeamAdapter physics backend.

Wraps stEVE's low-level SOFA scene graph construction into the clean
PhysicsBackend interface. The SOFA scene graph logic is preserved verbatim
from stEVE (the FEM physics are correct); only the API boundary changes.
"""
from __future__ import annotations
import logging
import math
import os
from typing import List, Optional, Tuple
import numpy as np

from .backend import PhysicsBackend
from ..core.types import DeviceState, DeviceSpec, AnatomySpec

logger = logging.getLogger(__name__)


def _quaternion_from_direction(direction: np.ndarray) -> List[float]:
    """Return quaternion [x, y, z, w] rotating X-axis to given direction."""
    d = direction / np.linalg.norm(direction)
    orig = np.array([1.0, 0.0, 0.0])
    if np.allclose(d, orig):
        return [0.0, 0.0, 0.0, 1.0]
    if np.allclose(d, -orig):
        return [0.0, 1.0, 0.0, 0.0]
    half = (orig + d) / np.linalg.norm(orig + d)
    w = float(np.dot(orig, half))
    xyz = np.cross(orig, half).tolist()
    return xyz + [w]


class SofaBackend(PhysicsBackend):
    """
    SOFA BeamAdapter FEM backend.

    Action space per device: [translation_mm_per_s, rotation_rad_per_s]
    """

    def __init__(self, dt_simulation: float = 0.006, friction: float = 0.1):
        self.dt_simulation = dt_simulation
        self.friction = friction

        self._sofa = None
        self._root = None
        self._instruments_combined = None
        self._dof_positions: np.ndarray = np.zeros((1, 3))
        self._sim_error = False
        self._rng = np.random.default_rng()
        self._n_devices = 0

    # ------------------------------------------------------------------ #
    # PhysicsBackend interface                                             #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        anatomy: AnatomySpec,
        devices: List[DeviceSpec],
        seed: Optional[int] = None,
    ) -> List[DeviceState]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._n_devices = len(devices)
        self._sim_error = False

        self._load_sofa()
        self._build_scene(anatomy, devices)
        self._update_dof_positions()
        return self._read_device_states()

    def step(
        self,
        actions: np.ndarray,
        dt: float,
    ) -> List[DeviceState]:
        n_substeps = max(1, int(round(dt / self.dt_simulation)))
        ctrl = self._instruments_combined.m_ircontroller

        for _ in range(n_substeps):
            x_tip = ctrl.xtip
            rot = ctrl.rotationInstrument
            for i in range(self._n_devices):
                x_tip[i] += float(actions[i, 0] * self.dt_simulation)
                rot[i] += float(actions[i, 1] * self.dt_simulation)
            ctrl.xtip = x_tip
            ctrl.rotationInstrument = rot
            try:
                self._sofa.Simulation.animate(self._root, self._root.dt.value)
            except Exception as exc:
                logger.warning("SOFA animation error: %s", exc)
                self._sim_error = True
                break

        self._update_dof_positions()
        return self._read_device_states()

    def get_dof_positions(self) -> np.ndarray:
        return self._dof_positions.copy()

    @property
    def simulation_error(self) -> bool:
        return self._sim_error

    def close(self) -> None:
        if self._root is not None and self._sofa is not None:
            try:
                self._sofa.Simulation.unload(self._root)
            except Exception:
                pass
        self._root = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_sofa(self):
        if self._sofa is not None:
            self.close()
        try:
            import Sofa
            import Sofa.Core
            import SofaRuntime
            self._sofa = Sofa
            self._sofa_runtime = SofaRuntime
        except ImportError as exc:
            raise RuntimeError(
                "SOFA not found. Set SOFA_ROOT and PYTHONPATH. "
                "See simulation/install_env.sh for setup instructions."
            ) from exc

    def _build_scene(self, anatomy: AnatomySpec, devices: List[DeviceSpec]):
        self._root = self._sofa.Core.Node("root")
        self._root.gravity = [0.0, 0.0, 0.0]
        self._root.dt = self.dt_simulation

        self._root.addObject("RequiredPlugin", pluginName=(
            "Sofa.Component.AnimationLoop "
            "Sofa.Component.Collision.Detection.Algorithm "
            "Sofa.Component.Collision.Detection.Intersection "
            "Sofa.Component.Collision.Geometry "
            "Sofa.Component.Collision.Response.Contact "
            "Sofa.Component.Constraint.Lagrangian.Correction "
            "Sofa.Component.Constraint.Lagrangian.Solver "
            "Sofa.Component.IO.Mesh "
            "Sofa.Component.LinearSolver.Direct "
            "Sofa.Component.Mapping.Linear "
            "Sofa.Component.Mass "
            "Sofa.Component.ODESolver.Backward "
            "Sofa.Component.SolidMechanics.Spring "
            "Sofa.Component.StateContainer "
            "Sofa.Component.Topology.Container.Dynamic "
            "Sofa.Component.Topology.Container.Grid "
            "Sofa.Component.Visual "
            "BeamAdapter"
        ))
        self._root.addObject("FreeMotionAnimationLoop")
        self._root.addObject(
            "GenericConstraintSolver",
            tolerance=1e-5,
            maxIterations=100,
        )
        self._root.addObject(
            "CollisionPipeline", draw=False, depth=6, verbose=False
        )
        self._root.addObject("BruteForceBroadPhase")
        self._root.addObject("BVHNarrowPhase")
        self._root.addObject(
            "CollisionResponse",
            response="FrictionContactConstraint",
            responseParams=f"mu={self.friction}",
        )
        self._root.addObject(
            "LocalMinDistance",
            name="Proximity",
            alarmDistance=3.0,
            contactDistance=1.0,
            angleCone=0.02,
        )

        self._add_vessel(anatomy)
        self._add_devices(anatomy, devices)

        self._sofa.Simulation.init(self._root)

    def _add_vessel(self, anatomy: AnatomySpec):
        vessel = self._root.addChild("Vessel")
        vessel.addObject("MeshObjLoader", filename=anatomy.mesh_path,
                         flipNormals=False, name="meshLoader")
        vessel.addObject("MeshTopology",
                         position="@meshLoader.position",
                         triangles="@meshLoader.triangles")
        vessel.addObject("MechanicalObject", name="dofs", src="@meshLoader")
        vessel.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel.addObject("LineCollisionModel", moving=False, simulated=False)

    def _sofa_device_params(self, spec: DeviceSpec) -> dict:
        """Convert DeviceSpec → WireRestShape kwargs."""
        if spec.kind == "j_guidewire":
            straight = spec.total_length - spec.tip_length
            return dict(
                isAProceduralShape=True,
                straightLength=straight,
                length=spec.total_length,
                spireDiameter=2.0 * spec.tip_length / math.pi * spec.tip_angle,
                radiusExtremity=spec.diameter / 2,
                youngModulusExtremity=spec.young_modulus * 0.5,
                massDensityExtremity=0.000155,
                radius=spec.diameter / 2,
                youngModulus=spec.young_modulus,
                massDensity=0.000155,
                poissonRatio=spec.poisson_ratio,
                keyPoints=[0.0, straight, spec.total_length],
                densityOfBeams=[40, 10],
                numEdgesCollis=[20, 10],
                numEdges=50,
                spireHeight=0.0,
            )
        elif spec.kind == "straight":
            return dict(
                isAProceduralShape=True,
                straightLength=spec.total_length,
                length=spec.total_length,
                spireDiameter=4.0,
                radiusExtremity=spec.diameter / 2,
                youngModulusExtremity=spec.young_modulus,
                massDensityExtremity=0.000155,
                radius=spec.diameter / 2,
                youngModulus=spec.young_modulus,
                massDensity=0.000155,
                poissonRatio=spec.poisson_ratio,
                keyPoints=[0.0, spec.total_length],
                densityOfBeams=[50],
                numEdgesCollis=[30],
                numEdges=50,
                spireHeight=0.0,
            )
        else:
            raise ValueError(f"Unknown device kind: {spec.kind!r}")

    def _add_devices(self, anatomy: AnatomySpec, specs: List[DeviceSpec]):
        nx = 0
        for i, spec in enumerate(specs):
            params = self._sofa_device_params(spec)
            topo = self._root.addChild(f"topolines_{i}")
            topo.addObject("WireRestShape", name=f"rest_{i}",
                           template="Rigid3d", printLog=False, **params)
            topo.addObject("EdgeSetTopologyContainer", name=f"meshLines_{i}")
            topo.addObject("EdgeSetTopologyModifier", name="Modifier")
            topo.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo",
                           template="Rigid3d")
            topo.addObject("MechanicalObject", name=f"dofTopo_{i}",
                           template="Rigid3d")
            nx += sum(params["densityOfBeams"])

        combined = self._root.addChild("InstrumentCombined")
        combined.addObject("EulerImplicitSolver",
                           rayleighStiffness=0.2, rayleighMass=0.1)
        combined.addObject("BTDLinearSolver",
                           verification=False, subpartSolve=False, verbose=False)
        combined.addObject("RegularGridTopology", name="MeshLines",
                           nx=nx + 1, ny=1, nz=1,
                           xmax=1.0, xmin=0.0, ymin=0, ymax=0,
                           zmax=1, zmin=1, p0=[0, 0, 0])
        combined.addObject("MechanicalObject", showIndices=False,
                           name="DOFs", template="Rigid3d")

        x_tip, rotations, interp_str = [], [], ""
        for i, spec in enumerate(specs):
            combined.addObject("WireBeamInterpolation",
                               name=f"Interpol_{i}",
                               WireRestShape=f"@../topolines_{i}/rest_{i}",
                               radius=spec.diameter / 2,
                               printLog=False)
            params = self._sofa_device_params(spec)
            combined.addObject("AdaptiveBeamForceFieldAndMass",
                               name=f"ForceField_{i}",
                               massDensity=params["massDensity"],
                               interpolation=f"@Interpol_{i}")
            x_tip.append(0.1 if i == 0 else 0.0)
            rotations.append(float(self._rng.random() * math.pi * 2))
            interp_str += f"Interpol_{i} "

        ip = np.array(anatomy.insertion_point)
        idir = np.array(anatomy.insertion_direction)
        quat = _quaternion_from_direction(idir)
        pose = ip.tolist() + quat[:3] + [quat[3]]

        combined.addObject("InterventionalRadiologyController",
                           name="m_ircontroller",
                           template="Rigid3d",
                           instruments=interp_str.strip(),
                           startingPos=pose,
                           xtip=x_tip,
                           rotationInstrument=rotations,
                           speed=0.0,
                           listening=True,
                           controlledInstrument=0,
                           printLog=False)
        combined.addObject("LinearSolverConstraintCorrection",
                           wire_optimization="true", printLog=False)
        combined.addObject("FixedConstraint", indices=0)
        combined.addObject("RestShapeSpringsForceField",
                           points="@m_ircontroller.indexFirstNode",
                           angularStiffness=1e8, stiffness=1e8,
                           external_points=0,
                           external_rest_shape="@DOFs")

        collis = combined.addChild("CollisionModel")
        collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        collis.addObject("EdgeSetTopologyModifier", name="collisEdgeModifier")
        collis.addObject("MechanicalObject", name="CollisionDOFs")
        collis.addObject("MultiAdaptiveBeamMapping",
                         controller="../m_ircontroller",
                         useCurvAbs=True, printLog=False, name="collisMap")
        collis.addObject("LineCollisionModel", proximity=0.0)
        collis.addObject("PointCollisionModel", proximity=0.0)

        self._instruments_combined = combined

    def _update_dof_positions(self):
        try:
            dofs = self._instruments_combined.DOFs
            pos = dofs.position.value
            self._dof_positions = np.array([p[:3] for p in pos], dtype=np.float32)
        except Exception as exc:
            logger.debug("Could not read DOF positions: %s", exc)

    def _read_device_states(self) -> List[DeviceState]:
        ctrl = self._instruments_combined.m_ircontroller
        states = []
        for i in range(self._n_devices):
            inserted = float(ctrl.xtip[i]) if hasattr(ctrl.xtip, "__len__") else 0.0
            rotation = float(ctrl.rotationInstrument[i]) \
                if hasattr(ctrl.rotationInstrument, "__len__") else 0.0
            tip = self._dof_positions[0].copy() if len(self._dof_positions) > 0 \
                else np.zeros(3)
            states.append(DeviceState(
                dof_positions=self._dof_positions,
                inserted_length=inserted,
                rotation=rotation,
                tip_position=tip,
            ))
        return states
