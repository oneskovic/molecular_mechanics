# Molecular mechanics
A university project at the Belgrade University of Mathematics, part of the Introduction to mechanics course.
Implements a molecular dynamics simulation using the **AMBER** force field and integration of the equations of motion using the **velocity Verlet** algorithm. Energy minimization done using the **Limited-memory BFGS (L-BFGS)** algorithm in PyTorch. The following forces are implemented: `HarmonicBondForce`, `HarmonicAngleForce`, `DihedralForce`, `LennardJonesForce`, `CoulombForce`.

![Dynamics simulation of the 1cfg protein ](https://github.com/oneskovic/molecular_mechanics/blob/main/animations/1cfg-dynamics.gif)