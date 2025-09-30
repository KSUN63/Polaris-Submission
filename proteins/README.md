# Protein Ensemble Processing Workflow

## Overview
Three-step workflow for preparing protein structures for docking studies. The clustered proteins from both experimental crystal structures and MCSCE-generated structured can be found in the folder `Protein_Ensemble`

## Steps

### 1. Protein Alignment (`alignment.py`)
- Aligns protein structures to reference using CA atoms of key residues
- Supports SARS-CoV-2 and MERS-CoV Mpro alignment
- Outputs aligned protein and ligand files

### 2. Clustering (`clustering.py`) 
- Performs DBSCAN clustering on aligned protein ensembles
- Uses side chain atom coordinates of key residues as features
- Selects representative structures from each cluster

### 3. Protein Fixing (`fix_protein.py`)
- Fixes missing atoms/residues using PDBFixer
- Cleans HETATM entries and standardizes hydrogen naming
- Prepares final structures for docking

## Key Residues (based on interactions)
- SARS-CoV-2: [41, 140-145, 163-166, 172, 187-189]
- MERS-CoV: [41, 143-148, 166-169, 175, 190-192]