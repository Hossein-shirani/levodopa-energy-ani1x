import torch
import torchani
from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import BFGS

class LevodopaOptimizer:
    def __init__(self, symbols, positions, dihedral_indices, dihedral_value=70.0):
        self.symbols = symbols
        self.positions = positions
        self.dihedral_indices = dihedral_indices
        self.dihedral_value = dihedral_value
        self.model = torchani.models.ANI1x(periodic_table_index=True).ase()
        self.atoms = self._build_atoms()
        self._apply_dihedral_constraint()

    def _build_atoms(self):
        atoms = Atoms(self.symbols, positions=self.positions)
        atoms.calc = self.model
        return atoms

    def _apply_dihedral_constraint(self):
        initial_dihedral = self.atoms.get_dihedral(*self.dihedral_indices)
        self.atoms.set_dihedral(*self.dihedral_indices, self.dihedral_value)
        constraint = FixInternals(dihedrals_deg=[[initial_dihedral, self.dihedral_indices]])
        self.atoms.set_constraint(constraint)

    def optimize(self, fmax=0.0001):
        optimizer = BFGS(self.atoms)
        optimizer.run(fmax=fmax)
        return self.atoms.get_potential_energy()

if __name__ == "__main__":
    symbols = 'CCCCCCHHHOHOHCHHCHCOHONHH'
    positions = [
        [-1.20470200,  1.04669100,  0.26250000],
        [-0.33574100, -0.02743200,  0.48551100],
        [-0.84692000, -1.31870700,  0.39783600],
        [-2.19238900, -1.52237200,  0.09385500],
        [-3.05023700, -0.45328500, -0.12481600],
        [-2.54558400,  0.85223500, -0.03894300],
        [-0.82472500,  2.06685200,  0.32783700],
        [-0.18721900, -2.16802000,  0.54518100],
        [-2.58479100, -2.53652600,  0.02508300],
        [-4.37261400, -0.59290800, -0.42227300],
        [-4.57886800, -1.53553900, -0.44733400],
        [-3.42015500,  1.87301800, -0.26082800],
        [-2.94289200,  2.70701200, -0.16828600],
        [ 1.11614000,  0.24576200,  0.80827500],
        [ 1.27565300,  1.32419900,  0.89777300],
        [ 1.38102000, -0.19348300,  1.78068300],
        [ 2.10161400, -0.29430300, -0.24020800],
        [ 1.67791000, -0.09208500, -1.23400000],
        [ 3.40080900,  0.51505800, -0.19580400],
        [ 4.49079300, -0.23877900, -0.45353800],
        [ 5.24402400,  0.37496400, -0.44120200],
        [ 3.47847700,  1.70324300,  0.00718700],
        [ 2.26565900, -1.73376100, -0.11776600],
        [ 2.85858600, -2.08253000, -0.86484600],
        [ 2.75174500, -1.94700500,  0.74984100]
    ]
    dihedral_indices = [2, 1, 13, 16]

    optimizer = LevodopaOptimizer(symbols, positions, dihedral_indices)
    final_energy = optimizer.optimize()
    print(f"Final energy after constrained optimization: {final_energy:.4f} eV")
