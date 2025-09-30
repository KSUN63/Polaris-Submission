import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# key residues for SARS-CoV-2 and MERS-CoV Mpro
key_residue_sars = [41, 140, 141, 142, 143, 144, 145, 163, 164, 165, 166, 172, 187, 188, 189]
key_residue_mers = [41, 143, 144, 145, 146, 147, 148, 166, 167, 168, 169, 175, 190, 191, 192]

# pymol alignment based on key residues and output transformation matrix
sars_ref = "ALIGNMENT_REFERENCES/SARS-CoV-2-Mpro/reference_structure/complex.pdb"
mers_ref = "ALIGNMENT_REFERENCES/MERS-CoV-Mpro/reference_structure/complex.pdb"