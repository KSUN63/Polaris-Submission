from pdbfixer import PDBFixer
from openmm.app import PDBFile

def fix_protein(input_pdb, output_pdb, keep_water=False, add_hydrogens=False, fix_id = True, keep_het = False):
    """
    Fix the protein structure using PDBFixer.

    Args:
        input_pdb: input pdb file
        output_pdb: output pdb file
    
    Returns:
        a pdb file with the missing atoms and residues added
    """
    # Load the PDB file using PDBFixer
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if not keep_het:
        fixer.removeHeterogens(keepWater=keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if add_hydrogens:
        fixer.addMissingHydrogens(7.4)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'), keepIds=fix_id)
    clean_protein(output_pdb, output_pdb)
    return output_pdb

def clean_protein(input_pdb, output_pdb):
    """
    Clean the protein structure after fixing or md relaxation.

    Args:
        input_pdb: input pdb file
        output_pdb: output pdb file
    
    Returns:
        a pdb file with the HETATM lines removed and H1 fixed
    """
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith('HETATM')]
    
    with open(output_pdb, 'w') as f:
        for line in lines:
            if line.startswith("ATOM") and int(line[22:26].strip()) == 1:
                if line[12:16].strip() == 'H':
                    # Change 'H' to 'H1' (right-aligned in a 4-character field)
                    line = line[:12] + ' H1 ' + line[16:]
            f.write(line)
    return output_pdb

if __name__ == "__main__":
    # example to fix the protein structure
    input_pdb = "SARS-CoV-2_Mpro-J0013_0A_CONFIDENTIAL/protein_aligned_sars.pdb"
    output_pdb = "SARS-CoV-2_Mpro-J0013_0A_CONFIDENTIAL/protein_aligned_sars_fixed.pdb"
    fix_protein(input_pdb, output_pdb)