import pymol, os
from rdkit import Chem
from Bio import PDB
from constants import key_residue_sars, key_residue_mers, sars_ref, mers_ref

# alignment based on chain A
def check_for_unk_chain(complex_path):
    """
    Returns the chain ID ('A' or 'B') whose His41 is closest to ligand atom 0
    
    Args:
        complex_path: Path to PDB complex with UNK ligand
    Returns:
        str: Chain ID ('A' or 'B') that's closest to ligand
    """
    # Load structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', complex_path)
    
    # Get His41 from both chains
    his41_A = structure[0]['A'][41]  # Model 0, Chain A, Residue 41
    his41_B = structure[0]['B'][41]  # Model 0, Chain B, Residue 41
    
    # Get first atom of UNK ligand
    for chain in structure[0]:
        for residue in chain:
            if residue.get_resname() == 'UNK':
                lig_atom0 = list(residue.get_atoms())[0]
                break
    
    # Calculate distances from ligand atom 0 to His41 CA of each chain
    dist_A = lig_atom0 - his41_A['CA']
    dist_B = lig_atom0 - his41_B['CA']
    
    # Return chain ID with smallest distance
    return 'A' if dist_A < dist_B else 'B'

def map_pdb_coordinates_to_sdf(pdb_path, sdf_path):
    mol_sdf = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mol_pdb = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    # get the coordinates of the atoms
    pdb_coords = mol_pdb.GetConformer().GetPositions()
    sdf_conf = mol_sdf.GetConformer()

    for i, coords in enumerate(pdb_coords):
        sdf_conf.SetAtomPosition(i, coords)

    writer = Chem.SDWriter(pdb_path.replace(".pdb", "_final.sdf"))
    writer.write(mol_sdf)
    writer.close()
    return pdb_path.replace(".pdb", "_final.sdf")

def align_pred(protein_path, self_type, ref_type):
    pymol.cmd.delete("all")
    if ref_type == 'MERS-CoV Mpro':
        ref_path = mers_ref
        name = "mers"
    elif ref_type == "SARS-CoV-2 Mpro":
        ref_path = sars_ref
        name = "sars"
    else:
        raise Exception
    chain = check_for_unk_chain(protein_path)
    pymol.cmd.load(protein_path, "mobile")
    pymol.cmd.load(ref_path, "reference")
    pymol.cmd.align(
        f"chain {chain} and mobile and name CA and resi {'+'.join(map(str, key_residue_sars if self_type=='SARS-CoV-2 Mpro' else key_residue_mers))}",
        f"chain A and reference and name CA and resi {'+'.join(map(str, key_residue_sars if ref_type=='SARS-CoV-2 Mpro' else key_residue_mers))}",
        transform=1,
        object="alignment_matrix"
    )
    ret_path = os.path.join(os.path.dirname(protein_path), f"protein_aligned_{name}.pdb")
    ligand_aligned = os.path.join(os.path.dirname(protein_path), f"ligand_aligned_{name}.pdb")
    pymol.cmd.save(ret_path, f"mobile and chain {chain}")
    pymol.cmd.save(ligand_aligned, f"mobile and resname UNK")
    
    map_pdb_coordinates_to_sdf(ligand_aligned, os.path.join(os.path.dirname(protein_path), "ligand.sdf"))
    
    return ret_path

if __name__ == "__main__":
    # example
    sars2_protein_folder = "SARS-CoV-2_Mpro-J0013_0A_CONFIDENTIAL"
    align_pred(os.path.join(sars2_protein_folder, "complex.pdb"), "SARS-CoV-2 Mpro", "SARS-CoV-2 Mpro")
    align_pred(os.path.join(sars2_protein_folder, "complex.pdb"), "SARS-CoV-2 Mpro", "MERS-CoV Mpro")