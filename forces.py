from torch import Tensor
import torch
from atom import Atom, get_bond_angle

class HarmonicBondForceParams:
    def __init__(self, l: float, k: float):
        self.l = l
        self.k = k

class HarmonicBondForce:
    def __init__(self, bond_dict: dict[tuple, HarmonicBondForceParams]):
        self.bond_dict = bond_dict
        # Add reverse bonds
        temp_dict = bond_dict.copy()
        for bond, params in temp_dict.items():
            self.bond_dict[(bond[1], bond[0])] = params

    def get_force(self, atom1: Atom, atom2: Atom) -> float:
        bond = (atom1.element, atom2.element)
        force = torch.tensor(0.0)
        if bond in self.bond_dict:
            l, k = self.bond_dict[bond].l, self.bond_dict[bond].k
            dist = (atom1.position - atom2.position).norm()
            force = k * (dist - l) ** 2 / 2
        return force

class HarmonicAngleForceParams:
    def __init__(self, angle: float, k: float):
        self.angle = angle
        self.k = k

class HarmonicAngleForce:
    def __init__(self, angle_dict: dict[tuple, HarmonicAngleForceParams]):
        self.angle_dict = angle_dict
        # Add reverse angles
        temp_dict = angle_dict.copy()
        for angle, params in temp_dict.items():
            self.angle_dict[(angle[2], angle[1], angle[0])] = params
    
    def get_force(self, atom1: Atom, atom2: Atom, atom3: Atom) -> float:
        atoms = (atom1.element, atom2.element, atom3.element)
        force = torch.tensor(0.0)
        if atoms in self.angle_dict:
            angle, k = self.angle_dict[atoms].angle, self.angle_dict[atoms].k
            current_angle = get_bond_angle(atom1, atom2, atom3)
            force = k * (current_angle - angle) ** 2 / 2
        return force

class LennardJonesForceParams:
    def __init__(self, epsilon: float, sigma: float):
        self.epsilon = epsilon
        self.sigma = sigma

class LennardJonesForce:
    def __init__(self, lj_dict: dict[str, LennardJonesForceParams]):
        self.lj_dict = lj_dict

    def get_force(self, atom1: Atom, atom2: Atom) -> float:
        force = torch.tensor(0.0)
        if atom1.element in self.lj_dict and atom2.element in self.lj_dict:
            epsilon1, sigma1 = self.lj_dict[atom1.element].epsilon, self.lj_dict[atom1.element].sigma
            epsilon2, sigma2 = self.lj_dict[atom2.element].epsilon, self.lj_dict[atom2.element].sigma
            dist = (atom1.position - atom2.position).norm()
            sigma = (sigma1 + sigma2) / 2
            epsilon = (epsilon1 * epsilon2) ** 0.5

            force = 4 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)
        return force
    
class CoulombForce:
    def __init__(self, charge_dict: dict[str, float]):
        self.charge_dict = charge_dict
    
    def get_force(self, atom1: Atom, atom2: Atom) -> float:
        force = torch.tensor(0.0)
        if atom1.element in self.charge_dict and atom2.element in self.charge_dict:
            charge1, charge2 = self.charge_dict[atom1.element], self.charge_dict[atom2.element]
            dist = (atom1.position - atom2.position).norm()
            k = 8.9875517873681764e9
            force = k * charge1 * charge2 / dist
        return force