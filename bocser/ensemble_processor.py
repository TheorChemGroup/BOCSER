import tensorflow as tf

from typing import List, Tuple, Callable#, Self

from calc import dihedral_angle, HARTRI_TO_KCAL

class EnsembleProcessor:
    def __init__(
        self,
        ensemble_filename : str,
        dihedral_idxs : List[List[int]],
        parse_energy_from_xyz_2nd_line : Callable[[str], float] = lambda s: float(s.split()[-1])*HARTRI_TO_KCAL
    ) -> None:
        """
            Processes .xyz ensemble to list of torsion angle values
            and their energies in kcal/mol. 

            Args:
                ensembel_filename - filename of ensemble
                dihedral_idxs - list of 4-element lists with atom idxs.
                    angles given in expected order
                parse_energy_from_xyz_2nd_line - function that will apply 
                    to the 2nd line of .xyz file and return Energy in kcal/mol
        """
        self.processed_ens = []
        xyz_blocks = []

        try:
            with open(ensemble_filename, 'r') as file:
                current_block = ""
                for idx, line in enumerate(file):
                    if len(line.split()) == 1 and idx > 0:
                        xyz_blocks.append(current_block)
                        current_block = ""
                    current_block += line
                xyz_blocks.append(current_block)
        except FileNotFoundError:
            print(f"Error! No such file: {ensemble_filename}; Finishing with empty ensemble!")
            return

        self.energies = [parse_energy_from_xyz_2nd_line(xyz_block.split('\n')[1]) for xyz_block in xyz_blocks]

        for raw_xyz in xyz_blocks:
            coords = [list(map(float, cur.split()[1:])) for cur in raw_xyz.split('\n')[2:]]
            if len(coords[-1]) == 0:
                coords.pop()
            dihedrals = []
            for i, j, k, l in dihedral_idxs:
                dihedrals.append(
                    dihedral_angle(
                        coords[i], coords[j], coords[k], coords[l]
                    )
                )
            self.processed_ens.append(dihedrals)

    def normalize_energy(
        self,
        norm_en : float
        ):# -> Self:
        """
            Substracts `norm_en` from all energies in ensemble
            Args:
                norm_en - normalization energy
            Returns:
                self object

        """

        self.energies = [cur - norm_en for cur in self.energies]
        return self
    

    def get_data(
        self    
    ) -> Tuple[List[List[float]], List[float]]:
        """
            Returns parsed data

            Return:
                tuple of list of dihedral values and list of energies 
        """
        return self.processed_ens, self.energies

    def get_tf_data(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            Returns parsed data in tf float64 tensors

            Return:
                tuple of tf tensor of torsion values and tf tensor of energies
        """

        return tf.constant(self.processed_ens, dtype=tf.float64), tf.reshape(tf.constant(self.energies, dtype=tf.float64), [-1, 1])
