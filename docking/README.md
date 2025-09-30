# VINA-GPU Docking Pipeline

This repository contains a Python pipeline for running AutoDock VINA-GPU molecular docking on server environments. The pipeline automates the preparation of protein and ligand files, configures docking parameters, and executes batch docking runs.

## Prerequisites

### Software Dependencies

Before running the docking pipeline, ensure you have the following software installed:

1. **AutoDock VINA-GPU 2.1**
   - Download from: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
   - Follow the compilation and installation instructions in the repository

2. **ADFRsuite**
   - Required for protein preparation (`prepare_receptor`)
   - Download from: https://ccsb.scripps.edu/adfr/

3. **Meeko**
   - Required for ligand preparation and conversion
   - Install via conda: `conda install -c conda-forge meeko`

4. **Python Dependencies**
   - RDKit
   - tqdm
   - pandas
   - numpy

### System Requirements

- Linux-based server environment
- GPU with CUDA support (for VINA-GPU)
- Sufficient disk space for input/output files
- Adequate memory for batch processing

## Configuration

### ⚠️ IMPORTANT: Update File Paths

Before running the pipeline, you **MUST** update the file paths in `dependency_path.py` to match your system installation:

```python
# CHANGE THESE UNDER DIFFERENT SYSTEMS
ADFR_install_path = '/path/to/your/ADFRsuite/bin'
meeko_install_path = '/path/to/your/conda/envs/vina/bin'

protein_prep_path = f'{ADFR_install_path}/prepare_receptor'
meeko_ligprep_path = f"{meeko_install_path}/mk_prepare_ligand.py"
meeko_ligconv_path = f"{meeko_install_path}/mk_export.py"

OPENCL_PATH = "/path/to/your/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"
VINA_GPU_BINARY_PATH = "/path/to/your/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"
LD_LIBRARY_PATH = "/path/to/your/envs/vina-gpu-env/lib"
```
## Input Data Structure

The pipeline expects the following directory structure:

```
vina-prep-frag/
├── sars_ligand1/
│   ├── ligand_smiles.smi          # SMILES strings of ligands
│   └── sars_protein.pdb           # Protein structure file
├── sars_ligand2/
│   ├── ligand_smiles.smi
│   └── sars_protein.pdb
├── mers_ligand1/
│   ├── ligand_smiles.smi
│   └── mers_protein.pdb
└── ...
```

## Usage

After completing protein alignment, you can perform docking using the provided script:

```bash
python run_docking.py
```

The script will:
1. Process all directories in the input path
2. Convert SMILES to 3D structures
3. Prepare protein and ligand files for docking
4. Generate configuration files
5. Execute VINA-GPU docking
6. Convert results back to SDF format
