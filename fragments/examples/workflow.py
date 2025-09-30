#!/usr/bin/env python3
"""
Fragment Analysis Workflow

This script demonstrates a complete workflow for fragment analysis using two different
fragmentation schemes: ring-based and MMPA. It processes training data from ligand
directories and test data from SMILES files, then performs similarity analysis using
fingerprint-based BallTree search.

The script automatically caches fragment generation results to avoid recomputation.
Cache files are stored in the 'fragment_cache/' directory.

Usage:
    python workflow.py

The script will:
1. Load training data from SARS-CoV-2 ligand directories
2. Generate fragments using both ring-based and MMPA methods (with caching)
3. Load test data from SMILES files
4. Create fingerprint BallTrees for similarity search
5. Query similar fragments for each test molecule
6. Compare fragment coverage between methods
"""

import os
import sys
import glob
import pickle
from typing import Dict, List, Tuple
import numpy as np
from rdkit import Chem

# Add parent directory to path to import fragments modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fragments.frag import MoleculeFragmenter
from fragments.utils import (
    parse_ligand_info, 
    convert_frag_to_fingerprint,
    create_fingerprint_balltree,
    query_similar_fragments,
    generate_morgan_fingerprint
)

def get_cache_filename(data_type: str, method: str, base_path: str = "../ligand-posing/") -> str:
    """
    Generate cache filename for fragment dictionaries.
    
    Args:
        data_type: Type of data ("train", "sars_test", "mers_test")
        method: Fragmentation method ("ring_based", "mmpa")
        base_path: Base path for training data
        
    Returns:
        Cache filename
    """
    cache_dir = "fragment_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    if data_type == "train":
        # Use a hash of the base path to make filename unique
        import hashlib
        path_hash = hashlib.md5(base_path.encode()).hexdigest()[:8]
        return os.path.join(cache_dir, f"train_fragments_{method}_{path_hash}.pkl")
    else:
        return os.path.join(cache_dir, f"{data_type}_fragments_{method}.pkl")

def save_fragments(frag_dict: Dict, filename: str):
    """
    Save fragment dictionary to pickle file.
    
    Args:
        frag_dict: Fragment dictionary to save
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(frag_dict, f)
    print(f"Saved fragments to {filename}")

def load_fragments(filename: str) -> Dict:
    """
    Load fragment dictionary from pickle file.
    
    Args:
        filename: Input filename
        
    Returns:
        Fragment dictionary
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            frag_dict = pickle.load(f)
        print(f"Loaded fragments from {filename}")
        return frag_dict
    return None

def load_training_data(base_path: str = "../ligand-posing/") -> List[str]:
    """
    Load training data directories for SARS-CoV-2 ligands.
    
    Args:
        base_path: Base path to ligand-posing directory
        
    Returns:
        List of directory paths
    """
    pattern = os.path.join(base_path, "SARS-CoV-2*")
    train_dirs = [file for file in glob.glob(pattern) if os.path.isdir(file)]
    print(f"Found {len(train_dirs)} training directories")
    return train_dirs

def generate_training_fragments(train_dirs: List[str], method: str = "ring_based", 
                              base_path: str = "../ligand-posing/", use_cache: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate fragments for training molecules using specified method with caching.
    
    Args:
        train_dirs: List of training directory paths
        method: Fragmentation method ("ring_based" or "mmpa")
        base_path: Base path for training data (used for cache filename)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dictionary mapping molecule IDs to fragment fingerprints
    """
    cache_filename = get_cache_filename("train", method, base_path)
    
    # Try to load from cache first
    if use_cache:
        cached_frag_dict = load_fragments(cache_filename)
        if cached_frag_dict is not None:
            print(f"Using cached training fragments for {method} method")
            return cached_frag_dict
    
    print(f"\nGenerating training fragments using {method} method...")
    
    fragmenter = MoleculeFragmenter(method)
    frag_dict = {}
    
    for folder in train_dirs:
        folder_name = os.path.basename(folder)
        print(f"Processing {folder_name}...")
        
        mol = parse_ligand_info(folder)
        if mol is not None:
            frags = fragmenter.fragment(mol)
            frag_dict[folder_name] = convert_frag_to_fingerprint(frags)
            print(f"  Generated {len(frags)} fragments")
        else:
            print(f"  Warning: Could not parse molecule from {folder}")
    
    print(f"Total training molecules processed: {len(frag_dict)}")
    
    # Save to cache
    if use_cache:
        save_fragments(frag_dict, cache_filename)
    
    return frag_dict

def load_test_smiles(file_path: str) -> Dict[str, Chem.Mol]:
    """
    Load test molecules from SMILES file.
    
    Args:
        file_path: Path to SMILES file
        
    Returns:
        Dictionary mapping SMILES to RDKit molecules
    """
    print(f"\nLoading test molecules from {file_path}...")
    
    molecules = {}
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            smiles = line.strip()
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules[smiles] = mol
                else:
                    print(f"  Warning: Invalid SMILES on line {line_num}: {smiles}")
    
    print(f"Loaded {len(molecules)} test molecules")
    return molecules

def generate_test_fragments(test_molecules: Dict[str, Chem.Mol], method: str = "ring_based", 
                           data_type: str = "test", use_cache: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate fragments for test molecules using specified method with caching.
    
    Args:
        test_molecules: Dictionary mapping SMILES to RDKit molecules
        method: Fragmentation method ("ring_based" or "mmpa")
        data_type: Type of test data ("sars_test", "mers_test", etc.)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dictionary mapping SMILES to fragment fingerprints
    """
    cache_filename = get_cache_filename(data_type, method)
    
    # Try to load from cache first
    if use_cache:
        cached_frag_dict = load_fragments(cache_filename)
        if cached_frag_dict is not None:
            print(f"Using cached {data_type} fragments for {method} method")
            return cached_frag_dict
    
    print(f"\nGenerating {data_type} fragments using {method} method...")
    
    fragmenter = MoleculeFragmenter(method)
    frag_dict = {}
    
    for smiles, mol in test_molecules.items():
        frags = fragmenter.fragment(mol)
        frag_dict[smiles] = convert_frag_to_fingerprint(frags)
        print(f"  {smiles}: {len(frags)} fragments")
    
    print(f"Total {data_type} molecules processed: {len(frag_dict)}")
    
    # Save to cache
    if use_cache:
        save_fragments(frag_dict, cache_filename)
    
    return frag_dict

def perform_similarity_analysis(train_frag_dict: Dict[str, Dict[str, np.ndarray]], 
                               test_frag_dict: Dict[str, Dict[str, np.ndarray]],
                               k: int = 5, 
                               similarity_threshold: float = 0.6) -> Dict:
    """
    Perform similarity analysis between training and test fragments.
    
    Args:
        train_frag_dict: Training fragment dictionary
        test_frag_dict: Test fragment dictionary
        k: Number of similar fragments to return
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Results dictionary containing similar fragments for each test molecule
    """
    print(f"\nPerforming similarity analysis...")
    print(f"Parameters: k={k}, similarity_threshold={similarity_threshold}")
    
    # Create BallTree from training fragments
    print("Creating BallTree from training fragments...")
    train_tree, train_identifiers, train_mapping = create_fingerprint_balltree(train_frag_dict)
    
    print(f"BallTree created with {len(train_identifiers)} training fragments")
    
    # Query similar fragments
    print("Querying similar fragments...")
    results = query_similar_fragments(
        test_frag_dict, 
        train_tree, 
        train_identifiers, 
        k=k, 
        similarity_threshold=similarity_threshold
    )
    
    # Print summary statistics
    total_queries = sum(len(mol_frags) for mol_frags in test_frag_dict.values())
    total_results = sum(len(frag_results) for mol_results in results.values() 
                       for frag_results in mol_results.values())
    
    print(f"Query summary:")
    print(f"  Total test fragments: {total_queries}")
    print(f"  Total similar fragments found: {total_results}")
    print(f"  Average results per fragment: {total_results/total_queries:.2f}")
    
    return results, train_tree, train_identifiers, train_mapping

def run_workflow(method: str = "ring_based", 
                base_path: str = "../ligand-posing/",
                sars_test_file: str = "../TEST_SMILES/sars2_polaris_test.smi",
                mers_test_file: str = "../TEST_SMILES/mers_polaris_test.smi",
                k: int = 5,
                similarity_threshold: float = 0.6,
                use_cache: bool = True):
    """
    Run the complete fragment analysis workflow.
    
    Args:
        method: Fragmentation method ("ring_based" or "mmpa")
        base_path: Base path to ligand-posing directory
        sars_test_file: Path to SARS test SMILES file
        mers_test_file: Path to MERS test SMILES file
        k: Number of similar fragments to return
        similarity_threshold: Minimum similarity threshold
    """
    print("=" * 80)
    print(f"FRAGMENT ANALYSIS WORKFLOW - {method.upper()} METHOD")
    print("=" * 80)
    
    # Step 1: Load training data
    train_dirs = load_training_data(base_path)
    if not train_dirs:
        print(f"Error: No training directories found in {base_path}")
        return
    
    # Step 2: Generate training fragments
    train_frag_dict = generate_training_fragments(train_dirs, method, base_path, use_cache)
    if not train_frag_dict:
        print("Error: No training fragments generated")
        return
    
    # Step 3: Load and process test data
    results_summary = {}
    
    # Process SARS test data
    if os.path.exists(sars_test_file):
        print(f"\n{'='*60}")
        print("PROCESSING SARS TEST DATA")
        print(f"{'='*60}")
        
        sars_test_molecules = load_test_smiles(sars_test_file)
        sars_test_frag_dict = generate_test_fragments(sars_test_molecules, method, "sars_test", use_cache)
        
        sars_results, sars_tree, sars_identifiers, sars_mapping = perform_similarity_analysis(
            train_frag_dict, sars_test_frag_dict, k, similarity_threshold
        )
        
        results_summary['sars'] = {
            'test_frag_dict': sars_test_frag_dict,
            'results': sars_results,
            'tree': sars_tree,
            'identifiers': sars_identifiers,
            'mapping': sars_mapping
        }
    else:
        print(f"Warning: SARS test file not found: {sars_test_file}")
    
    # Process MERS test data
    if os.path.exists(mers_test_file):
        print(f"\n{'='*60}")
        print("PROCESSING MERS TEST DATA")
        print(f"{'='*60}")
        
        mers_test_molecules = load_test_smiles(mers_test_file)
        mers_test_frag_dict = generate_test_fragments(mers_test_molecules, method, "mers_test", use_cache)
        
        mers_results, mers_tree, mers_identifiers, mers_mapping = perform_similarity_analysis(
            train_frag_dict, mers_test_frag_dict, k, similarity_threshold
        )
        
        results_summary['mers'] = {
            'test_frag_dict': mers_test_frag_dict,
            'results': mers_results,
            'tree': mers_tree,
            'identifiers': mers_identifiers,
            'mapping': mers_mapping
        }
    else:
        print(f"Warning: MERS test file not found: {mers_test_file}")
    
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return {
        'method': method,
        'train_frag_dict': train_frag_dict,
        'results_summary': results_summary
    }

def calculate_identical_fragment_coverage(test_frag_dict: Dict[str, Dict[str, np.ndarray]], 
                                        train_frag_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Calculate the percentage of test fragments that have identical matches in training set.
    
    Args:
        test_frag_dict: Test fragment dictionary
        train_frag_dict: Training fragment dictionary
        
    Returns:
        Dictionary with coverage statistics
    """
    # Create a set of all training fragment SMILES for fast lookup
    train_fragment_smiles = set()
    for mol_frags in train_frag_dict.values():
        train_fragment_smiles.update(mol_frags.keys())
    
    total_test_fragments = 0
    covered_test_fragments = 0
    
    for test_mol_id, test_frags in test_frag_dict.items():
        for test_frag_smiles in test_frags.keys():
            total_test_fragments += 1
            if test_frag_smiles in train_fragment_smiles:
                covered_test_fragments += 1
    
    coverage_percentage = (covered_test_fragments / total_test_fragments * 100) if total_test_fragments > 0 else 0
    
    return {
        'total_test_fragments': total_test_fragments,
        'covered_fragments': covered_test_fragments,
        'coverage_percentage': coverage_percentage,
        'unique_test_fragments': len(set().union(*[frags.keys() for frags in test_frag_dict.values()])),
        'unique_train_fragments': len(train_fragment_smiles)
    }

def compare_fragmentation_methods(use_cache: bool = True):
    """
    Compare results from both fragmentation methods with identical fragment coverage analysis.
    
    Args:
        use_cache: Whether to use cached results if available
    """
    print("\n" + "="*80)
    print("COMPARING FRAGMENTATION METHODS")
    print("="*80)
    
    # Run with ring-based method
    print("Running ring-based fragmentation...")
    ring_results = run_workflow(method="ring_based", use_cache=use_cache)
    
    # Run with MMPA method
    print("Running MMPA fragmentation...")
    mmpa_results = run_workflow(method="mmpa", use_cache=use_cache)
    
    # Compare results
    print(f"\n{'='*80}")
    print("FRAGMENTATION METHOD COMPARISON")
    print(f"{'='*80}")
    
    for dataset in ['sars', 'mers']:
        if dataset in ring_results['results_summary'] and dataset in mmpa_results['results_summary']:
            ring_data = ring_results['results_summary'][dataset]
            mmpa_data = mmpa_results['results_summary'][dataset]
            
            print(f"\n{dataset.upper()} DATASET:")
            print(f"{'='*50}")
            
            # Calculate identical fragment coverage
            ring_coverage = calculate_identical_fragment_coverage(
                ring_data['test_frag_dict'], 
                ring_results['train_frag_dict']
            )
            
            mmpa_coverage = calculate_identical_fragment_coverage(
                mmpa_data['test_frag_dict'], 
                mmpa_results['train_frag_dict']
            )
            
            print(f"\nRING-BASED METHOD:")
            print(f"  Test molecules: {len(ring_data['test_frag_dict'])}")
            print(f"  Total test fragments: {ring_coverage['total_test_fragments']}")
            print(f"  Unique test fragments: {ring_coverage['unique_test_fragments']}")
            print(f"  Identical matches found: {ring_coverage['covered_fragments']}")
            print(f"  Coverage: {ring_coverage['coverage_percentage']:.2f}%")
            print(f"  Unique training fragments: {ring_coverage['unique_train_fragments']}")
            
            print(f"\nMMPA METHOD:")
            print(f"  Test molecules: {len(mmpa_data['test_frag_dict'])}")
            print(f"  Total test fragments: {mmpa_coverage['total_test_fragments']}")
            print(f"  Unique test fragments: {mmpa_coverage['unique_test_fragments']}")
            print(f"  Identical matches found: {mmpa_coverage['covered_fragments']}")
            print(f"  Coverage: {mmpa_coverage['coverage_percentage']:.2f}%")
            print(f"  Unique training fragments: {mmpa_coverage['unique_train_fragments']}")
            
            print(f"\nCOMPARISON SUMMARY:")
            coverage_diff = mmpa_coverage['coverage_percentage'] - ring_coverage['coverage_percentage']
            if coverage_diff > 0:
                print(f"  MMPA has {coverage_diff:.2f}% HIGHER coverage than ring-based")
            elif coverage_diff < 0:
                print(f"  Ring-based has {abs(coverage_diff):.2f}% HIGHER coverage than MMPA")
            else:
                print(f"  Both methods have identical coverage")
            
            fragment_ratio_ring = ring_coverage['total_test_fragments'] / len(ring_data['test_frag_dict']) if len(ring_data['test_frag_dict']) > 0 else 0
            fragment_ratio_mmpa = mmpa_coverage['total_test_fragments'] / len(mmpa_data['test_frag_dict']) if len(mmpa_data['test_frag_dict']) > 0 else 0
            print(f"  Avg fragments per molecule - Ring-based: {fragment_ratio_ring:.2f}, MMPA: {fragment_ratio_mmpa:.2f}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    all_ring_coverage = []
    all_mmpa_coverage = []
    
    for dataset in ['sars', 'mers']:
        if dataset in ring_results['results_summary'] and dataset in mmpa_results['results_summary']:
            ring_coverage = calculate_identical_fragment_coverage(
                ring_results['results_summary'][dataset]['test_frag_dict'], 
                ring_results['train_frag_dict']
            )
            mmpa_coverage = calculate_identical_fragment_coverage(
                mmpa_results['results_summary'][dataset]['test_frag_dict'], 
                mmpa_results['train_frag_dict']
            )
            all_ring_coverage.append(ring_coverage['coverage_percentage'])
            all_mmpa_coverage.append(mmpa_coverage['coverage_percentage'])
    
    if all_ring_coverage and all_mmpa_coverage:
        avg_ring_coverage = sum(all_ring_coverage) / len(all_ring_coverage)
        avg_mmpa_coverage = sum(all_mmpa_coverage) / len(all_mmpa_coverage)
        
        print(f"Average coverage across all datasets:")
        print(f"  Ring-based method: {avg_ring_coverage:.2f}%")
        print(f"  MMPA method: {avg_mmpa_coverage:.2f}%")
        print(f"  Difference: {avg_mmpa_coverage - avg_ring_coverage:.2f}%")

def main():
    """
    Main function to run the workflow.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fragment Analysis Workflow")
    parser.add_argument("--method", choices=["ring_based", "mmpa"], default="ring_based",
                       help="Fragmentation method to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare both fragmentation methods")
    parser.add_argument("--base-path", default="../ligand-posing/",
                       help="Base path to ligand-posing directory")
    parser.add_argument("--sars-test", default="../TEST_SMILES/sars2_polaris_test.smi",
                       help="Path to SARS test SMILES file")
    parser.add_argument("--mers-test", default="../TEST_SMILES/mers_polaris_test.smi",
                       help="Path to MERS test SMILES file")
    parser.add_argument("-k", type=int, default=5,
                       help="Number of similar fragments to return")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Similarity threshold")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching and regenerate all fragments")
    
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    if args.compare:
        compare_fragmentation_methods(use_cache=use_cache)
    else:
        run_workflow(
            method=args.method,
            base_path=args.base_path,
            sars_test_file=args.sars_test,
            mers_test_file=args.mers_test,
            k=args.k,
            similarity_threshold=args.threshold,
            use_cache=use_cache
        )

if __name__ == "__main__":
    main()
