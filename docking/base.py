'''
Author: Jie Li
Date Created: Oct 21, 2022

This file defines the BaseDocking class with interfaces to be realized by different docking protocols
'''

from pathlib import Path
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures._base import TimeoutError
import numpy as np
import math, random, os, subprocess
from .dependency_path import *

import logging
from typing import Optional
import os


def init_logger(logname: Optional[os.PathLike] = None) -> logging.Logger:
    # logging
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file
    if logname is not None:
        handler = logging.FileHandler(str(logname))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def random_id():
    '''
    Generate a random ID
    '''
    return str(random.randint(0,65536))

def unpack_helper(func, args):
    '''
    Helper function to unpack arguments for multiprocessing
    '''
    return func(*args)
    
    
class BaseDocking:
    def __init__(self, protein_pdb, docking_box, temp_path=None, logger=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        :param logger: a logger object to log some results
        :param temp_path: str, path to the temporary directory
        '''
        self.protein_path = Path(protein_pdb).resolve()
        self.protein_folder = self.protein_path.parent
        self.protein_name = self.protein_path.stem
        self.logger = logger


    def dock(self, ligands, output_dir, single_job_timeout=120):
        '''
        Dock a list of ligands to the pocket in the protein

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param single_job_timeout: int, timeout for each job in seconds

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        pass

    def _get_parallel_docking_args(self, ligands, output_dir, single_job_timeout, n_jobs):
        zipped_args = zip(ligands, [output_dir] * len(ligands), [single_job_timeout] * len(ligands))
        return zipped_args


    def dock_parallel(self, ligands, output_dir, n_jobs=1, single_job_timeout=90, verbose=True, save_df_freq=500, **kwargs):
        '''
        Dock a list of ligands to the pocket in the protein in parallel

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param output_dir: str, path to the output directory
        :param n_jobs: int, number of jobs to run in parallel
        :param single_job_timeout: int, timeout for each job in seconds
        :param verbose: bool, whether to show progress bar
        :param save_df_freq: int, frequency to save the results to the disk (to prevent losing results)

        :return: pd.DataFrame with columns ["ligand_names", "smiles", "score", "path"], path is the path to the docked conformation
        '''
        
        pool = ProcessPoolExecutor(n_jobs)
        ligands = [[ligand] for ligand in ligands]
        zipped_args = self._get_parallel_docking_args(ligands, output_dir, single_job_timeout, n_jobs)
        counter = 0
        results = []
        if verbose:
            pbar = tqdm(total=len(ligands))
        futures = [pool.submit(self.dock, *args) for args in zipped_args]
        for ligand, future in zip(ligands, futures):
            try:
                result = future.result(timeout=single_job_timeout)
            except TimeoutError:
                result = pd.DataFrame({"ligand_names": [Path(ligand[0]).stem], "smiles": ["timeout"],
                               "vina_score": [np.nan], "vina_path": [""]})
            counter += 1
            results.append(result)
            if verbose:
                pbar.update(1)
            if counter % save_df_freq == 0:
                df = pd.concat(results)
                df.to_csv(Path(output_dir) / "results.csv", index=False)
                if self.logger is not None:
                    self.logger.info(f"Saved checkpoint results to {output_dir}/results.csv")

        final_results = pd.concat(results)
        return final_results.reset_index(drop=True)
        

    def rescore(self, ligands):
        '''
        Rescore given ligand conformations using the current docking protocol

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file

        :return: pd.DataFrame with columns ["index", "smiles", "score"]
        '''
        pass
    
    def rescore_parallel(self, ligands, n_jobs=1, n_chunks=5, single_job_timeout=500, verbose=True, **kwargs):
        '''
        Rescore a list of ligands in their provided poses with the protein in parallel

        :param ligands: list of ligands, each ligand is a path to the corresponding .sdf file
        :param n_jobs: int, number of jobs to run in parallel
        :param n_chunks: int, number of chunks to split the ligands into
        :param single_job_timeout: int, timeout for each job in seconds
        :param verbose: bool, whether to show progress bar
        :param save_df_freq: int, frequency to save the results to the disk (to prevent losing results)

        :return: pd.DataFrame with columns ["original_name", "smiles", "score", "path"], path is the path to the original conformation
        '''
        ligands = list(ligands) 
        pool = ProcessPoolExecutor(n_jobs)
        chunk_size = math.ceil(len(ligands) / n_chunks)
        ligands = [[ligands[i * chunk_size: (i + 1) * chunk_size]] for i in range(n_chunks)]

        counter = 0
        results = []
        if verbose:
            pbar = tqdm(total=n_chunks)
        futures = [pool.submit(self.rescore, *args) for args in ligands]
        for ligand, future in zip(ligands, futures):
            try:
                result = future.result(timeout=single_job_timeout)
            except TimeoutError:
                len_lig = len(ligand)
                result = pd.DataFrame({"ligand_name": ligand, "smiles": ["timeout"] * len_lig,
                               "vina_score": [np.nan] * len_lig, "vina_path": [""] * len_lig})
            counter += 1
            results.append(result)
            if verbose:
                pbar.update(1)
        del pool

        final_results = pd.concat(results)
        final_results.reset_index(drop=True)
        return final_results

    @staticmethod
    def convert_sdf_to_smiles(sdf_path):
        '''
        Convert a ligand sdf file to a smiles string, to be used by any of the docking protocols

        :param sdf_path: str, path to the sdf file

        :return: str, the smiles string
        '''
        sdmol = Chem.SDMolSupplier(sdf_path)
        mol = sdmol[0]
        if mol is None:
            # this is likely due to obabel conversion of pdbqt (only polar H)
            print("cannot read smiles from", sdf_path)
            return None
        try:
            return Chem.MolToSmiles(mol)
        except RuntimeError:
            return None


class AutoDockBaseDocking(BaseDocking):
    def __init__(self, protein_pdb, docking_box, temp_path=None, **kwargs) -> None:
        '''
        Initialize a docking protocol with a protein and a docking box

        :param protein_pdb: str, path to the protein pdb file
        :param docking_box: (xmin, ymin, zmin, xmax, ymax, zmax), the docking box definition
        '''
        super().__init__(protein_pdb, docking_box, temp_path, **kwargs)


    def convert_pdb_to_pdbqt(self, pdb_path, output_path, add_h = True):
        '''
        Convert a pdb file to a pdbqt file that can be used for AutoDock docking

        :param pdb_path: str, path to the pdb file
        :param output_path: str, path to the output pdbqt file

        :return: True if the run is successful
        '''
        protein_path = Path(pdb_path).resolve()
        protein_name = protein_path.stem
        protein_folder = protein_path.parent

        # preprocess the protein by removing water & heteroatoms
        # save the processed protein in the same folder as original pdb file
        with open(pdb_path, "r") as f:
            protein_file = f.read().split("\n")
        new_file = [i for i in protein_file if not i.startswith('HETATM')]
        processed_fp = os.path.join(protein_folder, "{}-processed.pdb".format(protein_name))
        with open(processed_fp, "w") as f1:
            f1.write("\n".join(new_file))
        
        # run protein preparation depending on if there's need to add H
        if add_h:
            try:
                out = subprocess.run([protein_prep_path, '-r', processed_fp, '-o', output_path,\
                '-A', 'checkhydrogens'])
            except subprocess.CalledProcessError as e:
                return e.output
        else:
            try:
                out = subprocess.run([protein_prep_path, '-r', processed_fp, '-o', output_path,])
            except subprocess.CalledProcessError as e:
                return e.output
        
        return True

    @staticmethod
    def convert_sdf_to_pdbqt(sdf_path, output_path):
        '''
        Convert a ligand sdf file to a pdbqt file that can be used for AutoDock docking

        :param sdf_path: str, path to the sdf file
        :param output_path: str, path to the output pdbqt file

        :return: True, if the run is successful
        '''
        try:
            out = subprocess.run([meeko_ligprep_path, '-i', sdf_path, '-o', output_path, "--rigid_macrocycles"])
        except subprocess.CalledProcessError as e:
            print('Bad molecule: '+ sdf_path)
            return False
        
        return True

    @staticmethod
    def convert_adresult_to_sdf(adresult_path, output_path, idx=-1):
        '''
        Convert a pdbqt/dlg file to a sdf file for standard file formatting

        :param adresult_path: str, path to the pdbqt/dlg file
        :param output_path: str, path to the output sdf file
        :param idx: int, the index of model to export

        :return: True, if the run is successful
        '''
        comment = ""
        if isinstance(idx, list): 
            comment = " ".join(idx)
            idx = -1
        temp_path = output_path if idx==-1 else f"/tmp/{random_id()}_{os.path.basename(output_path)}" 
        try:
            out = subprocess.run([meeko_ligconv_path, adresult_path, '-o', temp_path])
        except subprocess.CalledProcessError as e:
            return False
        if len(comment) > 0: 
            with open(temp_path, "a") as f:
                f.write("\n>  <REMARK>\nSELECTED MODELS: "+comment)
        if idx == -1: return True
        return AutoDockBaseDocking.extract_pose(temp_path, output_path, idx)
        
        
    @staticmethod
    def extract_pose(sdf_path, output_path, idx=0):
        '''
        extract the pose with idx in sdf containing multiple models
        
        :param sdf_path: str, path to the sdf result file
        :param output_path: str, path to the output sdf file
        :param idx: int, index of pdbqt model to extract

        :return: True, if the run is successful
        '''
        #assert not Path(sdf_path) == Path(output_path)
        with open(sdf_path, "r") as f:
            all_sdf = f.read().split("$$$$\n")
        # single model in pdbqt
        if len(all_sdf) == 1:
            with open(output_path, "w+") as f1:
                f1.write(all_sdf[0])
            return True
        elif len(all_sdf) - 1 <= idx:
            idx = 0
        # save the selected model
        with open(output_path, "w+") as f1:
            f1.write(all_sdf[idx])
        return True
    
    @staticmethod
    def read_smiles_from_pdbqt(pdbqt_path):
        '''
        Read a smiles string from pdbqt file, only for pdbqt generated by meeko
    
        :param pdbqt_path: str, path to the pdbqt file
    
        :return: str, the smiles string
        '''
        with open(pdbqt_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK SMILES'):
                    return line.strip().split()[-1]
    
    @staticmethod
    def read_energy_from_pdbqt(pdbqt_path):
        '''
        Read vina score from pdbqt file, only for pdbqt generated by meeko
    
        :param pdbqt_path: str, path to the pdbqt file
    
        :return: float, vina score
        '''
        with open(pdbqt_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK VINA RESULT:'):
                    return np.float(line.strip().split()[3])
    
    @staticmethod
    def read_smiles_from_dlg(dlg_path):
        '''
        Read a smiles string from a ligand dlg file, only for docking with pdbqt generated by meeko
    
        :param dlg_path: str, path to the dlg file
    
        :return: str, the smiles string
        '''
        with open(dlg_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('INPUT-LIGAND-PDBQT: REMARK SMILES'):
                    return line.strip().split()[-1]
