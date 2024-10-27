from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralRad

import numpy as np

from typing import Union
import os

from default_vals import ConfSearchConfig
from calc import start_calc, wait_for_the_end_of_calc
from coef_from_grid import calc_coefs
from db_connector import Connector

class CoefCalculator:
    """
        This class performs splitting given molecule on
        small parts with only one interesting rotable dihedral.
        Scanning energies of torsion rotation with given method.
        Calculating coefs for mean function for GPRegressor
    """

    def __init__(
        self,
        mol : Chem.rdchem.Mol,
        config : ConfSearchConfig,
        dir_for_inps : str="",
        skip_triple_equal_terminal_atoms=True,
        aromatic_to_aliphatic : bool = True,         
        degrees : np.ndarray = np.linspace(0, 2 * np.pi, 37).reshape(37, 1),
        db_connector : Union[Connector, None] = None
    ) -> None:
        """
            mol - rdkit molecule
            dir_for_inps - path to directory, where scan .inp files will generates
            skip_triple_equal_terminal_atoms - skip diherdrals,
                where one of atoms is RX3, where X - terminal atom
            num_of_procs - num of procs to calculate
            method_of_calc - method in orca format
            charge - charge of molecule
            multipl - multiplicity
            degrees - degree grid to scan
        """

        self.mol = mol
        self.dir_for_inps = dir_for_inps if dir_for_inps[-1] == "/" else dir_for_inps + "/"
        self.skip_triple_equal_terminal_atoms = skip_triple_equal_terminal_atoms
        self.num_of_procs = config.num_of_procs
        self.method_of_calc = config.orca_method
        self.charge = config.charge
        self.multipl = config.spin_multiplicity
        self.degrees = degrees

        # Key is SMILES, val is idx
        self.unique_frags = {}
        # Key is atom idxs, val is idx
        self.frags = {}

        self.db_connector = db_connector
        self.aromatic_to_aliphatic = aromatic_to_aliphatic

        self.case_sensetive_atoms = [
            cur for cur in [
                Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), idx) for idx in range(1, 119)
            ] if cur.upper() != cur
        ]

        self.scanfile2smiles = {} # k - scan_file, v - smiles
        self.fetched_coefs = {} # k - smiles, v - coefs

        if not os.path.exists(self.dir_for_inps):
            os.makedirs(self.dir_for_inps)

    def is_terminal(self,
                    atom : Chem.rdchem.Atom) -> bool:
        """
            Returns True if atom is terminal(Hs not counted)
        """
        return len(atom.GetNeighbors()) == 1

    def get_second_atom_in_bond(self,
                                bond : Chem.rdchem.Bond,
                                atom : Chem.rdchem.Atom) -> Chem.rdchem.Atom:
        """
            retruns another atom from this bond
        """
        return bond.GetEndAtom() if bond.GetBeginAtom().GetIdx() == atom.GetIdx() else bond.GetBeginAtom()

    def is_triple_eq_neighbors(self,
                               atom : Chem.rdchem.Atom) -> bool:
        """
        check if current atom has three equal neighbors

        """

        in_bond = None

        for bond in atom.GetBonds():
            if not self.is_terminal(self.get_second_atom_in_bond(bond, atom)):
                in_bond = bond
                break

        if in_bond is None:
            return False

        neighbor_atoms = [cur.GetSymbol() for cur in atom.GetNeighbors()]
        neighbor_atoms.remove(self.get_second_atom_in_bond(in_bond, atom).GetSymbol())

        neighbor_bonds = [cur.GetBondType() for cur in atom.GetBonds()]
        neighbor_bonds.remove(in_bond.GetBondType())

        terminal_neighbors = False

        # 3 terminal neighbors
        if sum([self.is_terminal(cur) for cur in atom.GetNeighbors()]) == 3:
            terminal_neighbors = True

        if(terminal_neighbors and len(neighbor_atoms) == 3 and len(set(neighbor_atoms)) == 1 and len(set(neighbor_bonds)) == 1):
            return True

        return False

    def is_interesting(self,
                       bond : Chem.rdchem.Atom) -> bool:
        """
            Returns True if we should scan this bond
            if skip_triple_equale_terminal_atoms == True - dihedral
            angels, where on one atom there are three equal terminal atoms,
            are not interesting
        """

        #If one of atoms is terminal
        if len([cur for cur in bond.GetBeginAtom().GetBonds()]) < 2 or\
           len([cur for cur in bond.GetEndAtom().GetBonds()]) < 2 :
            return False

        # If bond isn't single
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False

        if bond.IsInRing():
            return False

        if not self.skip_triple_equal_terminal_atoms:
            return True

        # If one of atoms has three equal terminal atom neighbors
        for t_atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if self.is_triple_eq_neighbors(t_atom):
                return False

        return True

    def get_unique_mols(
        self,
        lst : list[Chem.rdchem.Mol]
    ) -> list[Chem.rdchem.Mol]:
        """
            Return unqiue mols from lst. Leave first occurance only
        """

        occured_smiles = set()
        result = []
        
        for cur_mol in lst:
            cur_smiles = Chem.MolToSmiles(cur_mol)
            if cur_smiles in occured_smiles:
                continue
            result.append(cur_mol)
            occured_smiles.add(cur_smiles)

        return result

    def __get_unique_mols(self,
                        lst : list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
        """
            get unique molecules from list
        """

        return list(map(Chem.MolFromSmiles, set(list(map(Chem.MolToSmiles, lst)))))

    def generate_3d_coords(self,
                           lst : list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
        """
            returns list with same molecules but with
            Hs and generated coords by ETKDG
        """

        result = []

        for mol in lst:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            result.append(mol)

        return result

    def get_idxs_to_rotate(self,
                           mol : Chem.rdchem.Mol) -> list[int]:
        """
            Returns idxs of dihedral angel in correct order
        """

        for bond in mol.GetBonds():

            if not self.is_interesting(bond):
                continue

            return ([cur.GetIdx() for cur in bond.GetBeginAtom().GetNeighbors() if cur.GetIdx() != bond.GetEndAtomIdx()][0],
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    [cur.GetIdx() for cur in bond.GetEndAtom().GetNeighbors() if cur.GetIdx() != bond.GetBeginAtomIdx()][0])

    def convert_all_aromatic_to_aliphatic(
        self,
        cur_smiles : str
    ) -> str:
        """
            Converts all aromatic atoms in SMILES to aliphatic
        """
        tmp_smiles = cur_smiles
        for case_sensetive_atom in self.case_sensetive_atoms:
            if case_sensetive_atom in tmp_smiles:
                tmp_smiles = tmp_smiles.replace(case_sensetive_atom, f"<{case_sensetive_atom}>")
        counter = 0
        result = ""
        for cur in tmp_smiles:
            if cur == '<':
                counter += 1
            if cur == '>':
                counter -= 1
            if counter == 0 and cur.islower():
                result += cur.upper()
            else:
                result += cur
        for case_sensetive_atom in self.case_sensetive_atoms:
            if case_sensetive_atom in result:
                result = result.replace(f"<{case_sensetive_atom}>", case_sensetive_atom)
        return result

    def _sanitize_smiles(
        self,
        cur_smiles : str
    ) -> str:
        cur_mol = Chem.MolFromSmiles(cur_smiles)
        while True:
            print(f"Cur smiles: {Chem.MolToSmiles(cur_mol)}; Num of radical electrons: {sum([cur.GetNumRadicalElectrons() for cur in cur_mol.GetAtoms()])}")
            found_radical_electrons = False
            for atom in cur_mol.GetAtoms():
                found_radical_electrons |= atom.GetNumRadicalElectrons()
                atom.SetNumExplicitHs(atom.GetNumExplicitHs()+atom.GetNumRadicalElectrons())
            cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
            if not found_radical_electrons:
                break
            
        return Chem.MolToSmiles(Chem.RemoveAllHs(cur_mol))

    def get_interesting_frags(self) -> list[Chem.rdchem.Mol]:
        """
            returns a list of simple molecules with one
            rotable interesting torsion angle
            if skip_triple_equale_terminal_atoms == True - dihedral
            angels, where on one atom there are three equal terminal atoms,
            are not interesting
        """

        rotable_frags = []

        count = 0

        for bond in self.mol.GetBonds():
            if not self.is_interesting(bond):
                continue

            atoms_to_use = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

            for atom in [*bond.GetBeginAtom().GetNeighbors(),\
                         *bond.GetEndAtom().GetNeighbors()]:
                atoms_to_use.add(atom.GetIdx())

            if self.skip_triple_equal_terminal_atoms and\
               self.is_triple_eq_neighbors(atom):
                atoms_to_use.update([cur.GetIdx() for cur in atom.GetNeighbors()])

            rotable_frag_smiles = Chem.rdmolfiles.MolFragmentToSmiles(self.mol, atomsToUse = list(atoms_to_use))

            if not Chem.MolFromSmiles(rotable_frag_smiles):
                if self.aromatic_to_aliphatic:
                    rotable_frag_smiles = self.convert_all_aromatic_to_aliphatic(rotable_frag_smiles)
                else:
                    continue
            
            # Looks like shit, but works... I should rewrite it

            rotable_frag_smiles = self._sanitize_smiles(rotable_frag_smiles)

            rotable_frags.append(
                Chem.MolFromSmiles(
                    rotable_frag_smiles
                )
            )

            print(f"rot_frag_smiles: {rotable_frag_smiles}\nidxs_to_rotate: {self.get_idxs_to_rotate(rotable_frags[-1])}")

            query_result = self.mol.GetSubstructMatches(
                Chem.MolFromSmiles(
                    self._sanitize_smiles(
                        Chem.rdmolfiles.MolFragmentToSmiles(
                            rotable_frags[-1],
                            atomsToUse=self.get_idxs_to_rotate(rotable_frags[-1])
                        )
                    )
                )
            )
    
            print(f"query_result: {query_result}")

            old_idxs = ()

            for res in query_result:
                corr_idxs = True
                for cur in res:
                    corr_idxs = corr_idxs and cur in atoms_to_use
                if corr_idxs:
                    old_idxs = res
                    break

            frag_smiles = Chem.MolToSmiles(rotable_frags[-1])

            if frag_smiles in self.unique_frags:
                self.frags[old_idxs] = self.unique_frags[frag_smiles]
            else:
                self.unique_frags[frag_smiles] = count
                self.frags[old_idxs] = count
                count += 1

        return self.generate_3d_coords(self.get_unique_mols(rotable_frags))

    def get_list_of_xyz(self,
                        lst : list[Chem.rdchem.Mol]) -> list[str]:
        """
            returns list of xyz-blocks of given molecules
        """

        return list(map(Chem.MolToXYZBlock, lst))

    def generate_scan_inp(
        self,
        xyz : str,
        idxs_to_rotate : list[int],
        filename : str,
        submol_charge : int
    ) -> None:
        """
            Generates .inp file with "filename" for scan
            of mol, described by "xyz" xyz block, in orca
            Note that we rotate 0-1-2-3 angle, I think,
            that it should work always
        """
        with open(filename, 'w+') as file:
            file.write("!" + self.method_of_calc + " opt\n")
            file.write("%pal\nnprocs " + str(self.num_of_procs) + "\nend\n")
            file.write("%geom Scan\n")
            file.write("D " + " ".join(list(map(str, idxs_to_rotate))) + " = 0.0, 360.0, 37\n")
            file.write("end\nend\n")
            file.write("* xyz " + str(submol_charge) + " " + str(self.multipl) + "\n")
            file.write(xyz)
            file.write("END\n")

    def get_coords_from_xyz_block(self,
                                  xyz : str) -> str:
        """
            returns xyz-coords from xyz block
            erase first info lines
        """

        return "\n".join(xyz.split("\n")[2:])

    def generate_scan_inps_from_mol(self) -> list[str]:
        """
            Generates inp files for scanning of all interesting
            unique fragments from molecule.
            Returns list of .inp filenames
            dir_for_inps - path of directory including folders separator
        """

        inp_names = []

        angle_number = 0

        select_request = 'select a1, a2, a3, b1, b2, b3, c from dihedrals where ((dihedral_smiles = \"{smiles}\" and (method = \"{method}\")))'

        for sub_mol in self.get_interesting_frags():

            cur_mol = sub_mol
            cur_mol_smiles = Chem.MolToSmiles(Chem.RemoveHs(cur_mol))

            db_response = self.db_connector.get_request(
                select_request.format(
                    smiles=cur_mol_smiles,
                    method=self.method_of_calc.lower()
                )
            )
            if len(db_response) > 0:
                self.fetched_coefs[cur_mol_smiles] = db_response[0]

            SetDihedralRad(cur_mol.GetConformer(), 
                           *self.get_idxs_to_rotate(cur_mol),
                           0)
            xyz = Chem.MolToXYZBlock(cur_mol)
            idxs_to_rotate = self.get_idxs_to_rotate(cur_mol)
            filename = self.dir_for_inps + "scan_" + str(angle_number) + ".inp"
            self.generate_scan_inp(
                xyz=self.get_coords_from_xyz_block(xyz), 
                idxs_to_rotate=idxs_to_rotate, 
                filename=filename,
                submol_charge=Chem.GetFormalCharge(cur_mol)
            )
            inp_names.append(filename)
            angle_number += 1
            self.scanfile2smiles[filename] = cur_mol_smiles
    
        return inp_names

    def get_energies_from_scans(self,
                                lst : list[str]) -> list[tuple[str, list[float]]]:
        """
            lst - list of input file paths,
            return list of lists of energies in
            [0.0, 360.0] with step = 10 degrees
        """
        for inp_name in lst:
            out_name = inp_name[:-3] + "out"
            if not (self.scanfile2smiles[inp_name] in self.fetched_coefs):
                wait_for_the_end_of_calc(out_name, 1000)

        result = []

        for inp_name in lst:
            res_file_name = inp_name[:-3] + "relaxscanact.dat"

            if self.scanfile2smiles[inp_name] in self.fetched_coefs:
                result.append(None)
                continue

            cur_res = []
            with open(res_file_name, "r") as file:
                for line in file:
                    cur_res.append(float(line[:-1].split()[1]))
            result.append(np.array(cur_res))

        return zip(lst, result)

    def get_scans_of_dihedrals(self) -> np.ndarray:
        """
            Returns list of energie dependecies
        """

        inp_files = self.generate_scan_inps_from_mol()
        for cur in inp_files:
            if not (self.scanfile2smiles[cur] in self.fetched_coefs):
               start_calc(cur)    
        return self.get_energies_from_scans(inp_files)

    def calc(self) -> list[list[float]]:
        """
            Calculate coefs for mean function
        """
        res = []
        inp_filenames = []
        for inp_filename, energies in self.get_scans_of_dihedrals():
            inp_filenames.append(inp_filename)
            if self.scanfile2smiles[inp_filename] in self.fetched_coefs:
                res.append(self.fetched_coefs[self.scanfile2smiles[inp_filename]])
                continue
            res.append(calc_coefs(self.degrees, energies))
        
        print(f"Sucessful calculated {len(inp_filenames) - len(self.fetched_coefs)} coefs and fetched from db {len(self.fetched_coefs)} coefs!")    
        
        insert_request_template = 'insert into dihedrals (dihedral_smiles, method, a1, a2, a3, b1, b2, b3, c) values (\"{smiles}\", \"{method}\", {a1}, {a2}, {a3}, {b1}, {b2}, {b3}, {c})'
        
        for inp_filename, coefs in zip(inp_filenames, res):
            if self.scanfile2smiles[inp_filename] in self.fetched_coefs:
                continue
            self.db_connector.set_request(
                insert_request_template.format(**{
                    'smiles' : self.scanfile2smiles[inp_filename],
                    'method' : self.method_of_calc.lower(),
                    'a1' : coefs[0],
                    'a2' : coefs[1],
                    'a3' : coefs[2],
                    'b1' : coefs[3],
                    'b2' : coefs[4],
                    'b3' : coefs[5],
                    'c' : coefs[6]
                })
            )

        return res

    def coef_matrix(self) -> list[tuple[tuple, list[float]]]:
        """
            Get matrix of coefficients for mean function 
            for all dihedral angels 
        """  
        unique_coefs = self.calc()
        result = []        
        print(f"Frags: {self.frags}")

        for idxs in self.frags:
            result.append((list(idxs), unique_coefs[self.frags[idxs]]))
        
        print("DB Content:")
        for cur in self.db_connector.get_request('select * from dihedrals'):
            print(*cur, sep='|')

        return result
