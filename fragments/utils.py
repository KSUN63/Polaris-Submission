from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import os
from typing import Dict, List, Union

from .frag import MoleculeFragmenter
from sklearn.neighbors import BallTree
import numpy as np
from collections import defaultdict

def convert_frag_to_fingerprint(frags):
    """
    Convert the fragment to a fingerprint
    """
    return {Chem.MolToSmiles(frag): generate_morgan_fingerprint(frag) for frag in frags}

def generate_morgan_fingerprint(mol, radius=4, nBits=256):
    """
    Generate the Morgan fingerprint for the molecule
    """
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    feat = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, feat)
    return feat.astype(np.uint8)

def parse_ligand_info(folder):
    """
    Parse the ligand information from the folder
    """
    for file in os.listdir(folder):
        if file.endswith(".sdf"):
            mol = Chem.SDMolSupplier(os.path.join(folder, file))
            # Get the first molecule from the SDF file
            if mol and len(mol) > 0:
                mol = mol[0]
                if mol:
                    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return None

def get_frag_info(input_path: Union[str, List[str]], 
                  method: str = "ring_based") -> Dict[str, List[Chem.Mol]]:
    """
    Get fragment information from various input types using specified fragmentation method.
    
    Args:
        input_path: Path to directory, SMILES file, or list of directories/folders
        method: Fragmentation method ("ring_based" or "mmpa")
        fragmenter_params: Additional parameters for the fragmenter (e.g., {"max_cuts": 3})
    
    Returns:
        Dictionary mapping identifiers to fragment lists
    """
    # Initialize fragmenter with parameters
    fragmenter = MoleculeFragmenter(method)  
    frag_dict = {}
    
    # Handle list of directories/folders
    if isinstance(input_path, list):
        for folder in input_path:
            folder_name = os.path.basename(folder)
            mol = parse_ligand_info(folder)
            if mol is not None:
                frags = fragmenter.fragment(mol)
                frag_dict[folder_name] = convert_frag_to_fingerprint(frags)
    # Handle single directory
    elif os.path.isdir(input_path):
        for folder in os.listdir(input_path):
            folder_path = os.path.join(input_path, folder)
            if os.path.isdir(folder_path):
                mol = parse_ligand_info(folder_path)
                if mol is not None:
                    frags = fragmenter.fragment(mol)
                    frag_dict[folder] = convert_frag_to_fingerprint(frags)
    # Handle SMILES file
    elif input_path.endswith('.smi'):
        with open(input_path, "r") as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        frags = fragmenter.fragment(mol)
                        frag_dict[smiles] = convert_frag_to_fingerprint(frags)
    return frag_dict

def create_fingerprint_balltree(frag_dict):
    # Lists to store fingerprints and their corresponding identifiers
    all_fingerprints = []
    identifiers = []  # Will store (molecule_id, fragment_smiles) pairs
    mapping_pairs = defaultdict(list)
    
    # Flatten the nested dictionary structure
    for mol_id, fragments in frag_dict.items():
        for frag_smiles, fingerprint in fragments.items():
            all_fingerprints.append(fingerprint)
            identifiers.append([mol_id, frag_smiles])
            mapping_pairs[frag_smiles].append(mol_id)
    # Convert to numpy array
    X = np.vstack(all_fingerprints)  # Stack vertically since each fingerprint is already an array
    
    # Create BallTree with Jaccard metric
    tree = BallTree(X, metric='jaccard')
    
    return tree, identifiers, mapping_pairs

def search_similar_fragments(query_fingerprint, tree, identifiers, k=5, similarity_threshold=0.5):
    """
    Search for k most similar fragments to the query fingerprint
    
    Args:
        query_fingerprint: numpy array of the query fingerprint
        tree: BallTree object
        identifiers: list of (molecule_id, smiles) tuples
        k: number of neighbors to return
        similarity_threshold: minimum similarity score to include in results
    """
    # Reshape query to 2D array
    query = query_fingerprint.reshape(1, -1)
    
    # Find k nearest neighbors
    distances, indices = tree.query(query, k=k)
    
    # Get the corresponding identifiers
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 - dist  # Convert distance to similarity
        if similarity >= similarity_threshold:
            mol_id, frag_smiles = identifiers[idx]
            results.append({
                'molecule_id': mol_id,
                'fragment_smiles': frag_smiles,
                'similarity': similarity
            })
    
    return results

def query_similar_fragments(test_frag_dict, train_tree, train_identifiers, k=5, similarity_threshold=0.2):
    """
    Query similar fragments from training set for each fragment in test set
    
    Args:
        test_frag_dict: dictionary of test fragments
        train_tree: BallTree created from training fragments
        train_identifiers: list of (molecule_id, smiles) tuples for training data
        k: number of neighbors to return
        similarity_threshold: minimum similarity score to include
    """
    results = {}
    
    for test_mol_id, test_fragments in test_frag_dict.items():
        mol_results = {}
        for test_frag_smiles, test_fingerprint in test_fragments.items():
            # Search for similar fragments
            similar_frags = search_similar_fragments(
                test_fingerprint, 
                train_tree, 
                train_identifiers, 
                k=k, 
                similarity_threshold=similarity_threshold
            )
            
            # Store results
            mol_results[test_frag_smiles] = similar_frags
        
        results[test_mol_id] = mol_results
    
    return results