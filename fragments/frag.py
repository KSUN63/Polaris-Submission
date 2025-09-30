from rdkit import Chem
from rdkit.Chem import rdMMPA
from abc import ABC, abstractmethod
from typing import List, Optional

class FragmentationMethod(ABC):
    """Abstract base class for fragmentation methods."""
    
    @abstractmethod
    def fragment(self, molecule: Chem.Mol) -> List[Chem.Mol]:
        """Fragment a molecule into smaller fragments."""
        pass

class RingBasedFragmentation(FragmentationMethod):
    """Custom ring-based fragmentation method."""
    
    def fragment(self, molecule: Chem.Mol) -> List[Chem.Mol]:
        """Fragment molecule using ring-based approach."""
        return get_ring_and_connected_atoms(molecule)

class MMPAFragmentation(FragmentationMethod):
    """RDKit MMPA-based fragmentation method."""
    
    def __init__(self, max_cuts: int = 2, max_cut_bonds: int = 12):
        """
        Initialize MMPA fragmentation.
        
        Args:
            max_cuts: Maximum number of cuts to make
            max_cut_bonds: Maximum number of bonds to cut
        """
        self.max_cuts = max_cuts
        self.max_cut_bonds = max_cut_bonds
    
    def fragment(self, molecule: Chem.Mol) -> List[Chem.Mol]:
        """Fragment molecule using RDKit MMPA."""
        if molecule is None:
            return []
        
        try:
            # Get MMPA fragments - returns tuples when resultsAsMols=True
            frag_results = rdMMPA.FragmentMol(molecule, 
                                            maxCuts=self.max_cuts, 
                                            maxCutBonds=self.max_cut_bonds,
                                            resultsAsMols=True)
            
            # Extract molecules from the results
            fragments = []
            for frag_tuple in frag_results:
                if frag_tuple is not None:
                    # frag_tuple is (mol, cuts) where mol is the molecule
                    if isinstance(frag_tuple, tuple) and len(frag_tuple) > 0:
                        mol = frag_tuple[0]  # Extract the molecule from the tuple
                    else:
                        mol = frag_tuple  # Sometimes it might be just the molecule
                    
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            fragments.append(mol)
                        except Exception as e:
                            print(f"Warning: Could not sanitize MMPA fragment: {e}")
                            continue
            
            return fragments
        except Exception as e:
            print(f"Warning: MMPA fragmentation failed: {e}")
            return []

class MoleculeFragmenter:
    """Main class for molecule fragmentation with multiple methods."""
    
    def __init__(self, method: str = "ring_based"):
        """
        Initialize fragmenter with specified method.
        
        Args:
            method: Fragmentation method ("ring_based" or "mmpa")
        """
        self.method = method
        self._fragmenter = self._get_fragmenter()
    
    def _get_fragmenter(self) -> FragmentationMethod:
        """Get the appropriate fragmentation method."""
        if self.method == "ring_based":
            return RingBasedFragmentation()
        elif self.method == "mmpa":
            return MMPAFragmentation()
        else:
            raise ValueError(f"Unknown fragmentation method: {self.method}")
    
    def fragment(self, molecule: Chem.Mol) -> List[Chem.Mol]:
        """Fragment a molecule using the selected method."""
        return self._fragmenter.fragment(molecule)
    
    def set_method(self, method: str):
        """Change the fragmentation method."""
        self.method = method
        self._fragmenter = self._get_fragmenter()

def get_ring_and_connected_atoms(molecule):
    """
    Identify ring systems and expand them to include all connected atoms until another ring is encountered.
    """
    if molecule is None:
        return {}
    ring_info = molecule.GetRingInfo()
    ring_atoms = [set(ring) for ring in ring_info.AtomRings()]
    final_ring_atoms = ring_atoms[:]
    for i in range(len(ring_atoms)):
        for j in range(len(ring_atoms)):
            ring1, ring2 = ring_atoms[i], ring_atoms[j]
            if len(ring1.intersection(ring2)) > 0:
                final_ring_atoms[i] = final_ring_atoms[i].union(ring2)
    ring_atoms = list(set(frozenset(s) for s in final_ring_atoms))
    # Create a set of all ring atoms
    all_ring_atoms = set()
    for ring in ring_atoms:
        all_ring_atoms.update(ring)

    # Expand each ring using BFS to include connected atoms until another ring is encountered
    expanded_rings = []
    for ring in ring_atoms:
        expanded_ring = set(ring)
        visited = set(ring)

        # Use a stack for DFS
        stack = list(ring)

        while stack:
            current_atom_idx = stack.pop()
            current_atom = molecule.GetAtomWithIdx(current_atom_idx)

            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    if neighbor_idx not in all_ring_atoms:
                        expanded_ring.add(neighbor_idx)
                        stack.append(neighbor_idx)

        expanded_rings.append(expanded_ring)
    frags = []
    for ring in expanded_rings:
        atom_indices = list(ring)
        # Create an editable molecule
        fragment = Chem.RWMol()
        
        # Create a mapping of original indices to new indices
        idx_map = {}
        
        # First, add all atoms
        for orig_idx in atom_indices:
            orig_atom = molecule.GetAtomWithIdx(orig_idx)
            new_atom = Chem.Atom(orig_atom.GetSymbol())
            # Copy atom properties
            new_atom.SetFormalCharge(orig_atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(orig_atom.GetNumExplicitHs())
            # Add atom and store mapping
            new_idx = fragment.AddAtom(new_atom)
            idx_map[orig_idx] = new_idx
        
        # Then add all bonds between these atoms
        for orig_idx in atom_indices:
            atom = molecule.GetAtomWithIdx(orig_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in atom_indices and neighbor_idx > orig_idx:
                    bond = molecule.GetBondBetweenAtoms(orig_idx, neighbor_idx)
                    fragment.AddBond(idx_map[orig_idx], 
                                   idx_map[neighbor_idx], 
                                   bond.GetBondType())
        
        try:
            mol_fragment = fragment.GetMol()
            Chem.SanitizeMol(mol_fragment)
            frags.append(mol_fragment)
        except Exception as e:
            print(f"Warning: Could not sanitize fragment: {e}")
            continue
    return frags