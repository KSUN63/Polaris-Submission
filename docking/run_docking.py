from .base import AutoDockBaseDocking
from .base import BaseProject
from .pathlib import *
from pathlib import Path
import os, glob, shutil, re
from tqdm import tqdm

sars_box = (-20.46, -13.1, -32.79, 3.54, 10.9, -8.79)
mers_box = (-3.91, -12.03, 9.95, 20.09, 11.97, 33.95)

def write_config(receptor, lig_dir, output_dir, docking_box, config_fp):
    '''
    Write the config file for AutoDock Vina docking

    :param num_modes: int, the number of modes (conformations) to be generated
    :param energy_range: int, the energy range of the docking

    '''
    lines = [f"receptor = {receptor}",
        f"ligand_directory = {lig_dir}",
        f"output_directory = {output_dir}",
        "",
        "center_x = {}".format((docking_box[0] + docking_box[3]) / 2),
        "center_y = {}".format((docking_box[1] + docking_box[4]) / 2),
        "center_z = {}".format((docking_box[2] + docking_box[5]) / 2),
        "",
        "size_x = {}".format(docking_box[3] - docking_box[0]),
        "size_y = {}".format(docking_box[4] - docking_box[1]),
        "size_z = {}".format(docking_box[5] - docking_box[2]),
        "",
        "thread = 8000",
        "opencl_binary_path = {}".format(OPENCL_PATH)
    ]
    with open(config_fp, "w") as f:
        f.write("\n".join(lines))


def dock_folder(folder, type = "sars"):
    
    docking_box = sars_box if type == "sars" else mers_box
    all_config_path = []

    # first step is to convert ligand
    os.makedirs(os.path.join(folder, "docking"), exist_ok=True)
    proj = BaseProject('ligand_conversion', project_path = os.path.join(folder, "docking"))
    with open(os.path.join(folder, "ligand_smiles.smi"), "r") as f:
        smiles = [line.strip() for line in f.readlines()]
    names = [f"frag_dock_lig{i+1}" for i in range(len(smiles))]
    proj.add_multiple_ligands(smiles, names, format='smiles')
    
    # second step prepare protein and ligand into pdbqt
    proteins = glob.glob(os.path.join(folder, "*_protein.pdb"))
    if len(proteins) == 0: return []
    docking = AutoDockBaseDocking(proteins[0], "")
    lig_folder = os.path.join(folder, "docking", "ligands")
    protein_folder = os.path.join(folder, "docking", "proteins")
    vina_results = os.path.join(folder, "docking", "vina_results")
    os.makedirs(vina_results, exist_ok=True)
    ligand_path = glob.glob(os.path.join(lig_folder, '*.sdf')) # should be just 1 smiles
    for sdf_p in tqdm(ligand_path):
        docking.convert_sdf_to_pdbqt(sdf_p, sdf_p.replace(".sdf", "_in.pdbqt"))
    for pr in proteins:
        vina_docking_name =  pr.split("/")[-1].replace(".pdb", "")
        prepared_pdbqt_path = os.path.join(protein_folder, pr.split("/")[-1].replace(".pdb", ".pdbqt"))
        result_folder = os.path.join(vina_results, vina_docking_name)
        os.makedirs(result_folder, exist_ok=True)
        docking.convert_pdb_to_pdbqt(pr, prepared_pdbqt_path)
        config_fp = os.path.join(vina_results, f"{vina_docking_name}.txt")
        write_config(prepared_pdbqt_path, lig_folder, result_folder, docking_box, config_fp)
        all_config_path.append(config_fp)
    
    return all_config_path


def main(path_to_vina_prep_frag):

    path_to_vina_gpu_binary = os.path.join(path_to_vina_prep_frag, "run_vina_gpu.sh")
    # config everything
    config_paths_everything = []

    for folder in tqdm(glob.glob(path_to_vina_prep_frag + "/*")):
        if os.path.isdir(folder):
            name = os.path.basename(folder)
            protein_type = name[:4]
            configs = dock_folder(folder, protein_type)
            config_paths_everything.extend(configs)

    # write bash to run all the docking

    with open(path_to_vina_gpu_binary, "w") as f:
        f.write(f"export LD_LIBRARY_PATH={LD_LIBRARY_PATH}:$LD_LIBRARY_PATH"+"\n")
        f.write("ulimit -s 8192" + "\n")
        f.write(f"cd {VINA_GPU_BINARY_PATH}"+"\n")
        for config in config_paths_everything:
            if "protein.txt" in config:
                f.write(f"./AutoDock-Vina-GPU-2-1 --config {config}" + "\n")

    # then run the bash

    os.system(f"bash {path_to_vina_gpu_binary}")

    # convert docked result back to sdf

    folder = "vina-results-frag"
    os.makedirs(folder, exist_ok = True)
    results = glob.glob(os.path.join(path_to_vina_prep_frag, "*", "docking", "vina_results", "*", "*_out.pdbqt"))
    for output_fp in tqdm(results):
        protein = output_fp.split("/")[-2].replace("_protein", "")
        category, lig_name = output_fp.split("/")[-5].split("_")
        save_folder = os.path.join(folder, category, lig_name)
        os.makedirs(save_folder, exist_ok = True)
        AutoDockBaseDocking.convert_adresult_to_sdf(output_fp, os.path.join(save_folder, f"{protein}_docked.sdf"))

if __name__ == "__main__":
    path_to_vina_prep_frag = "/global/scratch/users/kysun/polaris-challenge/docking/vina-prep-frag"
    main(path_to_vina_prep_frag)