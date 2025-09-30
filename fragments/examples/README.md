# Fragment Analysis Examples

This directory contains example scripts demonstrating how to use the fragment analysis workflow.

## Files

- `workflow.py` - Complete workflow with command-line interface
- `simple_example.py` - Simple example replicating notebook functionality
- `README.md` - This file

## Quick Start

### Simple Example

To run the simple example that replicates your notebook:

```bash
cd fragments/examples
python simple_example.py
```

This will:
1. Load training data from `../ligand-posing/SARS-CoV-2*` directories
2. Generate fragments using ring-based method
3. Load test data from SMILES files
4. Create BallTree for similarity search
5. Query similar fragments

### Complete Workflow

To run the complete workflow with command-line options:

```bash
cd fragments/examples

# Run with ring-based method
python workflow.py --method ring_based

# Run with MMPA method
python workflow.py --method mmpa

# Compare both methods
python workflow.py --compare

# Custom parameters
python workflow.py --method mmpa -k 10 --threshold 0.5
```

## Command-line Options

- `--method`: Choose fragmentation method (`ring_based` or `mmpa`)
- `--compare`: Compare both fragmentation methods
- `--base-path`: Base path to ligand-posing directory (default: `../ligand-posing/`)
- `--sars-test`: Path to SARS test SMILES file (default: `../TEST_SMILES/sars2_polaris_test.smi`)
- `--mers-test`: Path to MERS test SMILES file (default: `../TEST_SMILES/mers_polaris_test.smi`)
- `-k`: Number of similar fragments to return (default: 5)
- `--threshold`: Similarity threshold (default: 0.6)

## Expected Directory Structure

```
fragments/
├── examples/
│   ├── workflow.py
│   ├── simple_example.py
│   └── README.md
├── frag.py
└── utils.py
../ligand-posing/
├── SARS-CoV-2_xxxxx/
├── SARS-CoV-2_xxxxx/
└── ...
../TEST_SMILES/
├── sars2_polaris_test.smi
└── mers_polaris_test.smi
```

## Output

The workflow generates:

1. **Training fragments**: Dictionary mapping molecule IDs to fragment fingerprints
2. **Test fragments**: Dictionary mapping SMILES to fragment fingerprints  
3. **BallTree**: For efficient similarity search
4. **Similarity results**: For each test fragment, list of similar training fragments

## Example Usage in Code

```python
from fragments.utils import get_frag_info, create_fingerprint_balltree, query_similar_fragments

# Load training data
train_dirs = glob.glob("../ligand-posing/SARS-CoV-2*")
train_frag_dict = get_frag_info(train_dirs, method="ring_based")

# Load test data
test_frag_dict = get_frag_info("../TEST_SMILES/sars2_polaris_test.smi", method="ring_based")

# Create BallTree
tree, identifiers, mapping = create_fingerprint_balltree(train_frag_dict)

# Query similar fragments
results = query_similar_fragments(test_frag_dict, tree, identifiers, k=5, similarity_threshold=0.6)
```

## Notes

- The ring-based method uses your custom fragmentation algorithm
- The MMPA method uses RDKit's rdMMPA for fragmentation
- Both methods generate Morgan fingerprints for similarity comparison
- The workflow handles missing files gracefully with warnings

