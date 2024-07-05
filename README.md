# Molecular mechanics
A university project at the Belgrade University of Mathematics, part of the Introduction to mechanics course.
Implements a molecular dynamics simulation using the [AMBER](https://ambermd.org/AmberModels.php) force field and integration of the equations of motion using the [Velocity Verlet](https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet) algorithm. Energy minimization done using the [Limited-memory BFGS (L-BFGS)](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html) algorithm in PyTorch. The following forces are implemented: [`HarmonicBondForce`](#harmonicbondforce), [`HarmonicAngleForce`](#harmonicangleforce), [`DihedralForce`](#dihedralforce), [`LennardJonesForce`](#lennardjonesforce), [`CoulombForce`](#coulombforce).

<p align="center">
<img src="https://github.com/oneskovic/molecular_mechanics/blob/main/animations/1cfg-dynamics.gif" alt="Dynamics simulation of the 1cfg protein " width="504" height="684" />
</p>

## Usage
### Install the libraries
```
pip install -r requirements.txt
```
### Run the simulation
Given a starting configuration of molecules (input.pdb), the following script generates a molecular trajectory file (output.xyz), i.e. a file containing snapshots of the system as it evolves over time. This trajectory can then be visualized using any third-party [molecular graphics system](https://en.wikipedia.org/wiki/List_of_molecular_graphics_systems). We recommend [VMD](https://www.ks.uiuc.edu/Research/vmd/).
```
python -m molecular_mechanics input.pdb output.xyz
```
Example of running the simulation on the [1cfg protein](https://www.rcsb.org/structure/1cfg) at 300K for 10000 iterations, upto 100 minimization iterations and saving the energy plot to figure.png:
```
python -m molecular_mechanics data/proteins/1cfg_h.pdb trajectory.xyz -it 10000 -mit 100 -plt figure.png -t 300 -fast
```
The input and output files are required. The only supported input formats are .pdb. Some example pdbs can be found in the [`data`](data/) folder.
Here's a table of all arguments supported by the simulator:
| Flag | Description | Default|
| ---- | ----------- | -------|
| **input file** | The input .pdb file describing the molecule to be simulated | None |
| **output file** | The output file to which the simulation result/s will be written | None |
| -t (--temperature) | The starting temperature (used for velocity initialization) | 293.15 |
| -it (--iterations) | The number of iteration simulations to run | 1000 |
|-plt (--save-energy-plot) | Specify this argument to save the energy plot | False |
|-v (--verbose) | Specify this argument to get verbose system state output during the simulation | False |
|-m (--minimize-energy) | Specify this argument to run energy minimization | False |
|-mit (--minimize-iterations) | The max number of minimization iterations to run (will possibly be less if L-BFGS converges) | 1000 |
|-ff (--force-field) | The xml of the force field to be used for atom parameters etc. | data/ff14SB.xml |
|-fast (--fast) | Whether to use the vectorized form of the simulation (runs on the gpu if possible using PyTorch) - **recommended** | False |
|-dt (--timestep) | Timestep used in the Verlet integration | 0.002 |

## Forces
Based on whether the forces act upon atoms that are considered "bonded" or not the forces belong to either bonded or nonbonded forces. The bonds between atoms are calculated from the input .pdb file on the start of the simulation. If the atom distance is less than some threshold the atoms are considered bonded and the bonds don't change during the simulation.

### Non-bonded forces
#### CoulombForce
The standard electrostatic force between two charged particles. For two atoms having charges $q_1$, $q_2$ separated by distance $d$ the force equals: $k\frac{q_1 q_2}{d}$, where $k$ is the coulomb constant.
#### LennardJonesForce
The Lennard-Jones potential (an approximation of the Van der Waals force). For two atoms separated by distance $d$, having parameters $\sigma_1$, $\epsilon_1$ and $\sigma_2$, $\epsilon_2$ the force equals: $4 \epsilon ((\frac{\sigma} {d})^{12} - (\frac{\sigma}{d})^6)$, where $\epsilon = \sqrt{\epsilon_1 \epsilon_2}$ and $\sigma = \frac{\sigma_1 + \sigma_2}{2}$.

### Bonded forces
#### HarmonicBondForce
The force that acts between two atoms that are bonded. For two atoms separated by distance $d$ with parameters $k$, $d_0$ the force equals: $k\frac{(d - d_0)^2}{2}$. The parameters $k$ and $d_0$ represent the bond strength and the equilibrium bond length.
#### HarmonicAngleForce
Similar to the HarmonicBondForce, the HarmonicAngleForce acts between three atoms that are bonded. For three atoms separated by angles $\theta$ with parameters $k$, $\theta_0$ the force equals: $k\frac{(\theta - \theta_0)^2}{2}$. The parameters $k$ and $\theta_0$ represent the force strength and the equilibrium angle.
#### DihedralForce
The force that acts between four atoms that are bonded. The angle formed between the two planes defined by the first three and the last three atoms is calculated this is $\theta$. The parameters $k$, $\theta_0$ and $n$ represent the force strength, $\theta_0$ is the phase offset and $n$ is the periodicity of the force. The force equals: $k(1 + \cos(n \theta - \theta_0))$.

## Callbacks
Can be attached to energy minimization or the dynamics simulation to get the current state of the system. The callbacks are called at the end of each iteration. The callback class should inherit from base `Callback`. The `__call__` function should accept the current iteration and state of the system as an argument. Check [`callbacks.py`](molecular_mechanics/callbacks.py) for some examples and some existing callbacks.
 
