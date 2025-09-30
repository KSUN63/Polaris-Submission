import glob, os, shutil, pickle
from rdkit import Chem
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit.Chem.rdFMCS import FindMCS

def select_ligand_from_multiple_references_simplified(folder_path, file_name = "candidate_ligands", weights = None):
    """
    Simplified selection method when all candidates are the same molecule in different poses.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing candidate ligand poses (.sdf files)
    reference_files : list
        List of paths to reference ligand SDF files
    weights : list, optional
        List of weights for each reference (defaults to equal weights)
        
    Returns:
    --------
    str : Path to best ligand pose
    """
    # Load all candidate poses
    ligand_files = glob.glob(os.path.join(folder_path, file_name, "*.sdf"))
    reference_files = glob.glob(os.path.join(folder_path, "*.sdf"))
    if not ligand_files or not reference_files:
        return None
    
    candidate_mols = []
    for ligand_file in ligand_files:
        mol = Chem.SDMolSupplier(ligand_file, removeHs=True)[0]
        if mol is not None:
            candidate_mols.append((ligand_file, mol))
    
    if not candidate_mols:
        return None
    
    # Use the first candidate to calculate MCS with references (since they're all the same molecule)
    template_mol = candidate_mols[0][1]
    
    # Load reference ligands
    references = []
    for i, ref_file in enumerate(reference_files):
        ref_mol = Chem.SDMolSupplier(ref_file, removeHs=True)[0]
        if ref_mol is None:
            continue
        # Assign weight
        weight = 1.0 if weights is None else weights[i]
        references.append((ref_mol, weight))
    # Pre-calculate MCS for each reference against the template molecule
    mcs_cache = []
    for ref_mol, weight in references:
        # Find maximum common substructure
        mcs = FindMCS([ref_mol, template_mol])
        if mcs.numAtoms < 5:  # Skip if MCS is too small
            continue
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        # Get atom indices for the MCS
        ref_match = ref_mol.GetSubstructMatch(mcs_mol)
        if not ref_match:
            continue
        mcs_cache.append((ref_mol, weight, mcs_mol, ref_match))
    
    # Score each candidate pose
    score_dict = {}
    
    for ligand_file, candidate_mol in candidate_mols:
        total_score = 0
        total_weight = 0
        
        for ref_mol, weight, mcs_mol, ref_match in mcs_cache:
            # Get atom indices for the candidate
            candidate_match = candidate_mol.GetSubstructMatch(mcs_mol)
            if not candidate_match:
                continue
            # Get coordinates
            ref_coords = ref_mol.GetConformer().GetPositions()[list(ref_match)]
            candidate_coords = candidate_mol.GetConformer().GetPositions()[list(candidate_match)]
            # Calculate RMSD of the matching fragment
            rmsd = np.sqrt(np.sum((ref_coords - candidate_coords) ** 2))
            # Weight the RMSD by the fragment size and reference importance
            fragment_fraction = len(ref_match) / ref_mol.GetNumAtoms()
            weighted_score = rmsd * weight * fragment_fraction
            
            total_score += weighted_score
            total_weight += weight * fragment_fraction
        
        # Calculate normalized score
        if total_weight > 0:
            final_score = total_score / total_weight
            score_dict[ligand_file] = final_score
    
    return score_dict