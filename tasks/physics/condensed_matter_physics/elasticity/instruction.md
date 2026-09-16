# First-Principles Mechanical Response of an Anisotropic Crystal

You are given a ground-state VASP POSCAR and DFT total energies for several small homogeneous strains of an orthorhombic crystal. Build a reproducible analysis pipeline that determines the crystal's second-order mechanical response and derived polycrystalline and acoustic properties.

## Inputs
- `/app/data/POSCAR_UNITCELL` VASP POSCAR containing lattice vectors, species, and atomic counts.
- `/app/data/DFT_ENERGY_STRAINS.csv` strain-pattern labels, strain amplitudes, and total energies.

## Required output
Create `/app/output/elastic_properties.json` containing, at minimum, the equilibrium cell volume, mass density, atom count, full 6x6 stiffness matrix, compliance matrix, Voigt/Reuss/Hill bulk and shear moduli, Young's modulus, Poisson ratio, Pugh ratio, universal anisotropy, directional mechanical properties, acoustic velocities, acoustic anisotropy, Debye temperature, fit diagnostics, and deterministic uncertainty estimates.

The analysis must use the supplied crystal symmetry and strain-energy data consistently. It must account for the distinction between normal and engineering shear strains, use the complete orthorhombic stiffness representation, and preserve SI units in reported derived quantities. The reported matrices and scalar properties must be mutually consistent, finite, and physically admissible. Directional quantities must be evaluated from the reconstructed fourth-rank stiffness tensor, and acoustic quantities must be obtained from the Christoffel formulation using the calculated density. The output must be generated from the supplied inputs rather than from embedded reference values.
