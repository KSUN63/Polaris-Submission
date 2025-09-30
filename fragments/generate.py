from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS

def find_mcs(smiles_1, smiles_2):
    mol_1 = Chem.MolFromSmiles(smiles_1)
    mol_2 = Chem.MolFromSmiles(smiles_2)
    mcs_result = FindMCS([mol_1, mol_2])
    return mcs_result.smartsString

def generate_constrained_conformers(mol_to_align, ref_mol, mcs_smarts, n_conformers=10):
    """
    Rapidly generate multiple conformers while maintaining core constraint
    """
    # Get atom mapping indices from MCS
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    align_match = mol_to_align.GetSubstructMatch(mcs_mol)
    
    if not ref_match or not align_match:
        raise ValueError("MCS matching failed.")
    
    # Create coordinate map from reference structure
    match_map = list(zip(ref_match, align_match))
    coordMap = {}

    for ref_idx, align_idx in match_map:
        ref_pos = ref_mol.GetConformer().GetAtomPosition(ref_idx)
        coordMap[align_idx] = ref_pos
    
    # Generate multiple conformers with coordinate constraints
    cids = AllChem.EmbedMultipleConfs(
        mol_to_align,
        numConfs=n_conformers,
        coordMap=coordMap,
        randomSeed=42,
        numThreads=0  # Use all available threads
    )
    
    if len(cids) == 0:
        raise ValueError("Failed to generate any conformers")
    
    # Minimize each conformer with constraints
    results = []
    for cid in cids:
        # # Create force field
        # ff = AllChem.MMFFGetMoleculeForceField(
        #     mol_to_align,
        #     AllChem.MMFFGetMoleculeProperties(mol_to_align),
        #     confId=cid
        # )
        
        # # Add positional constraints for mapped atoms
        # for align_idx in align_match:
        #     ff.AddFixedPoint(align_idx)
        
        # try:
        #     # Minimize with constraints
        #     ff.Minimize(maxIts=200)
        #     energy = ff.CalcEnergy()
        # except Exception as e:
        #     print(f"Minimization failed for conformer {cid}: {e}")
        #     continue
        
        # Align molecule and save the aligned coordinates to the current conformer
        rmsd = AllChem.AlignMol(mol_to_align, ref_mol, atomMap = list(zip(align_match, ref_match)), prbCid=cid)

        results.append((cid, rmsd))
    # Sort by energy
    results.sort(key=lambda x: x[1])
    
    return mol_to_align, results

def align_smiles_to_sdf(smiles: str, sdf_path: str, mcs_smarts: str, save_folder: str, n_conformers=10, save_name=None):
    """
    Aligns a molecule and generates multiple conformers
    """
    # Load reference molecule
    ref_supplier = Chem.SDMolSupplier(sdf_path)
    ref_mol = ref_supplier[0]
    
    if ref_mol is None:
        raise ValueError("Failed to read reference molecule from SDF.")

    # Convert input SMILES
    mol_to_align = Chem.MolFromSmiles(smiles)
    if mol_to_align is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens
    ref_mol = Chem.AddHs(ref_mol)
    mol_to_align = Chem.AddHs(mol_to_align)
    
    # Generate conformers
    mol_with_confs, results = generate_constrained_conformers(
        mol_to_align,
        ref_mol,
        mcs_smarts,
        n_conformers=n_conformers
    )
    
    # Save conformers
    for i, (conf_id, rmsd) in enumerate(results):
        save_path = os.path.join(save_folder, f"{save_name}_conf{i}.sdf")
        writer = Chem.SDWriter(save_path)
        
        # Create a new molecule with just this conformer
        conf_mol = Chem.Mol(mol_with_confs)
        conf_mol.RemoveAllConformers()
        conf_mol.AddConformer(mol_with_confs.GetConformer(conf_id))
        
        # Add properties
        conf_mol.SetProp('RMSD', str(rmsd))
        conf_mol.SetProp('ConfId', str(i))
        
        writer.write(conf_mol)
        writer.close()
    
    return mol_with_confs