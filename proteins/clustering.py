import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Bio import PDB
import glob, os, shutil
from tqdm import tqdm
from constants import key_residue_sars, key_residue_mers

# Function to extract side chain atom coordinates
def extract_side_chain_features(protein_files, key_residues):
    features = []
    parser = PDB.PDBParser(QUIET=True)
    for file in tqdm(protein_files):
        structure = parser.get_structure('protein', file)
        model = structure[0]  # Assuming single model
        feature_vector = []
        for chain in model:
            for residue in chain:
                if residue.id[1] in key_residues:
                    for atom in residue:
                        # exclude hydrogens
                        if not atom.get_name().startswith("H"):
                            feature_vector.extend(atom.get_coord())
        features.append(feature_vector)
    return features

# Assuming 'features' is your feature matrix
def perform_dbscan(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels

# Visualize clusters using PCA
def visualize_dbscan_clusters(features, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            label = 'Noise'
        else:
            label = f'Cluster {k}'

        class_member_mask = (labels == k)
        xy = reduced_features[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6, label=label)

    plt.title('DBSCAN Clustering Visualization with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    # example to extract features and perform clustering for SARS-CoV-2
    sars_protein_files = glob.glob("*/protein_aligned.pdb") # List of aligned protein file paths
    sars_features = extract_side_chain_features(sars_protein_files, key_residue_sars)

    counts = np.unique([len(f) for f in sars_features], return_counts=True)
    most_common_count = counts[0][np.argmax(counts[1])]
    print(most_common_count)

    all_index_to_remove = []
    for i, f in enumerate(sars_features):
        if len(f) != most_common_count:
            all_index_to_remove.append(i)

    sars_protein_files = [sars_protein_files[i] for i in range(len(sars_protein_files)) if i not in all_index_to_remove]
    sars_features = [sars_features[i] for i in range(len(sars_features)) if i not in all_index_to_remove]

    labels = perform_dbscan(sars_features, eps=5, min_samples=10)
    visualize_dbscan_clusters(sars_features, labels)

    rep_proteins = {}
    for i in range(len(labels)):
        if labels[i] == -1:
            rep_proteins['noise'] = sars_protein_files[i]
        else:
            rep_proteins[labels[i]] = sars_protein_files[i]
    rep_proteins

    for k,v in rep_proteins.items():
        if k != 'noise':
            shutil.copy(v, os.path.join("rep_proteins", f"cluster_{k}.pdb"))