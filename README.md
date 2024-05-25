# Molecular mechanics
A university project at the Belgrade University of Mathematics, part of the Introduction to mechanics course.
Implements a molecular dynamics simulation using the **AMBER** force field and integration of the equations of motion using the **velocity Verlet** algorithm. Energy minimization done using the **Limited-memory BFGS (L-BFGS)** algorithm in PyTorch. The following forces are implemented: `HarmonicBondForce`, `HarmonicAngleForce`, `DihedralForce`, `LennardJonesForce`, `CoulombForce`.

<p align="center">
<img src="https://github.com/oneskovic/molecular_mechanics/blob/main/animations/1cfg-dynamics.gif" alt="Dynamics simulation of the 1cfg protein " width="504" height="684" />
</p>

# Usage
### Install the libraries
```
pip install -r requirements.txt
```
### Run the simulation
```
python -m molecular_mechanics input.pdb output.xyz -t 273
```
The input and output files are required as well as the starting temperature (in kelvins). The only supported input formats are .pdb. Some example pdbs can be found in the data folder.
Here's a table of all arguments supported by the simulator:
| Flag | Description | Default|
| ---- | ----------- | -------|
| **input file** | The input .pdb file describing the molecule to be simulated | None |
| **output file** | The output file to which the simulation result/s will be written | None |
| **-t (--temperature)** | The starting temperature (used for velocity initialization) | None |
| -it (--iterations) | The number of iteration simulations to run | 1000 |
|-plt (--save-energy-plot) | Specify this argument to save the energy plot | False |
|-v (--verbose) | Specify this argument to get verbose system state output during the simulation | False |
|-m (--minimize-energy) | Specify this argument to run energy minimization | False |
|-mit (--minimize-iterations) | The max number of minimization iterations to run (will possibly be less if L-BFGS converges) | 1000 |
|-ff (--force-field) | The xml of the force field to be used for atom parameters etc. | data/ff14SB.xml |
|-fast (--fast) | Whether to use the vectorized form of the simulation (runs on the gpu if possible using PyTorch) - **recommended** | False |
|-dt (--timestep) | Timestep used in the Verlet integration | 0.002 |

# Forces
Based on whether the forces act upon atoms that are considered "bonded" or not the forces belong to either bonded or nonbonded forces. The bonds between atoms are calculated from the input .pdb file on the start of the simulation. If the atom distance is less than some threshold the atoms are considered bonded and the bonds don't change during the simulation.

## Non-bonded forces
### CoulombForce
The standard electrostatic force between two charged particles. For two particles having charges $q_1$, $q_2$ separated by distance $d$ the force equals: $k\frac{q_1 q_2}{d}$. Where $k$ is the coulomb constant.
### 
