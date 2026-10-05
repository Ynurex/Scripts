# =============================================================================
# contacts_ndx.py — Contactos residuo-residuo entre dos grupos de un .ndx
#
# Uso:
#   python contacts_ndx.py index.ndx --gro_file conf.gro --traj_file traj.xtc \
#          --group_A "Protein" --group_B "Ligand" [--distance_cutoff 0.6]
#
# Argumentos:
#   ndx_file           Índice de GROMACS con los grupos
#   --gro_file         Estructura (.gro)
#   --traj_file        Trayectoria (.xtc/.trr)
#   --group_A/B        Nombres de los grupos tal como aparecen en el .ndx
#   --distance_cutoff  Distancia mínima entre átomos para contar contacto (nm, def. 0.6)
#   --output           CSV de salida (def. contacts_analysis.csv)
#
# Salida:
#   - CSV con el tiempo de contacto (ps) para cada par de residuos A × B
#   - Por pantalla: tiempo total (ns) en que A y B están en contacto
# =============================================================================

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_ndx(ndx_file):
    """
    Load a GROMACS index file and return a dictionary of groups.

    Parameters
    ----------
    ndx_file : str
        Path to the GROMACS index file.

    Returns
    -------
    dict
        A dictionary where keys are group names and values are lists of atom indices.
    """
    groups = {}
    with open(ndx_file, 'r') as f:
        current_group = None
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_group = line[1:-1].strip()
                groups[current_group] = []
            elif current_group is not None:
                indices = list(map(int, line.split()))
                groups[current_group].extend(indices)
    return groups

def argparse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze contacts from a GROMACS index file.")
    parser.add_argument("--ndx_file", type=str, help="Path to the GROMACS index file.")
    parser.add_argument("--output", type=str, default="contacts_analysis.csv", help="Output CSV file for contact analysis.")
    parser.add_argument("--group_A", type=str, required=True, help="Name of the first group in the index file.")
    parser.add_argument("--group_B", type=str, required=True, help="Name of the second group in the index file.")
    parser.add_argument("--distance_cutoff", type=float, default=0.6, help="Distance cutoff for contact analysis (in nm).")
    parser.add_argument("--gro_file", type=str, required=True, help="Path to the GROMACS .gro file for the structure.") 
    parser.add_argument("--traj_file", type=str, required=True, help="Path to the GROMACS trajectory file for the simulation.")
    return parser.parse_args()

def main():
    args = argparse_arguments()
    groups = load_ndx(args.ndx_file)
    u = mda.Universe(args.gro_file, args.traj_file)

    if args.group_A not in groups or args.group_B not in groups:
        raise ValueError(f"One or both specified groups '{args.group_A}' and '{args.group_B}' not found in the index file.")
    
    group_A = u.atoms[np.array(groups[args.group_A]) - 1]
    group_B = u.atoms[np.array(groups[args.group_B]) - 1]

    print("Atoms in group A:", len(group_A))
    print("Atoms in group B:", len(group_B))

    residues_A = group_A.residues
    residues_B = group_B.residues

    print("Residues in group A:", len(residues_A))
    print("Residues in group B:", len(residues_B))

    # Calculate distances between atoms in the two groups
    num_frames = len(u.trajectory)
    times = np.zeros(num_frames)
    contact_matrix = np.zeros((num_frames, len(residues_A), len(residues_B)), dtype=bool)
    for i, ts in enumerate(u.trajectory):
        times[i] = ts.time  # ps

        for ia, resA in enumerate(residues_A):
            for ib, resB in enumerate(residues_B):
                d = distance_array(resA.atoms.positions,
                                resB.atoms.positions,
                                box=ts.dimensions)

                # distance_array usa Å → convertimos cutoff_nm → Å
                if d.min() <= args.distance_cutoff * 10:
                    contact_matrix[i, ia, ib] = True

    dt_ps = np.mean(np.diff(times))  # ps

    contact_frames = contact_matrix.sum(axis=0)
    contact_time_ps = contact_frames * dt_ps  # ps
    df_contacts = pd.DataFrame(contact_time_ps, index=[f"{resA.resname}{resA.resid}" for resA in residues_A],
                               columns=[f"{resB.resname}{resB.resid}" for resB in residues_B])
    df_contacts.to_csv(args.output)

    contact_AB = contact_matrix.any(axis=(1,2))

    total_contact_time_ns = contact_AB.sum() * dt_ps / 1000
    print(f"Tiempo total de interacción A–B: {total_contact_time_ns:.3f} ns")


if __name__ == "__main__":
    main()
