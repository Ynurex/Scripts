#!/usr/bin/env python3
"""
reorder_ligand_by_mol2.py
=========================

Reescribe el ligando dentro de un PDB de complejo (proteína + ligando) para que
sus átomos queden en EXACTAMENTE el mismo orden (y con los mismos nombres) que en
el mol2 con el que se generó el .itp, conservando las coordenadas del complejo.

Por qué: PyMOL/ChimeraX reordenan (y a veces renombran) los átomos al escribir el
PDB; grompp exige que el orden de los átomos en el .gro/.pdb coincida con el
[ atoms ] del .itp.

Estrategia de emparejamiento (PDB <-> mol2):
  1. "name":  por nombre de átomo (si todos los nombres coinciden 1:1 y la
              geometría es consistente con los enlaces del mol2).
  2. "graph": por isomorfismo de grafos moleculares (elemento + conectividad).
              La conectividad del mol2 sale de @<TRIPOS>BOND; la del PDB se
              infiere por distancias (radios covalentes). Funciona aunque el
              visor haya renombrado los hidrógenos. Entre mapeos equivalentes
              por simetría se elige el que más nombres conserva.
  "auto" (default) intenta 1 y si falla usa 2.

Salidas:
  --out        complejo PDB con el ligando reordenado (proteína intacta)
  --out-lig    (opcional) ligando solo, en orden del mol2, .pdb o .gro
               (útil para el flujo típico: pdb2gmx a la proteína + ligando aparte)

Dependencias: numpy, networkx  (pip install numpy networkx)
"""

import argparse
import sys
from collections import Counter, OrderedDict

import numpy as np

try:
    import networkx as nx
    from networkx.algorithms import isomorphism as iso
except ImportError:  # sólo es necesario para el modo "graph"
    nx = None

__version__ = "1.0"

# ----------------------------------------------------------------------------
# Datos de elementos
# ----------------------------------------------------------------------------
COV_RADII = {  # Å (Cordero et al. 2008, redondeados)
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Si": 1.11,
    "Se": 1.20, "Na": 1.66, "K": 2.03, "Mg": 1.41, "Ca": 1.76, "Zn": 1.22,
    "Fe": 1.32, "Cu": 1.32, "Mn": 1.39, "Co": 1.26, "Ni": 1.24,
}
MASSES = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904,
    "I": 126.90, "Si": 28.085, "Se": 78.97, "Na": 22.99, "K": 39.098,
    "Mg": 24.305, "Ca": 40.078, "Zn": 65.38, "Fe": 55.845, "Cu": 63.546,
    "Mn": 54.938, "Co": 58.933, "Ni": 58.693,
}
TWO_LETTER = {e for e in COV_RADII if len(e) == 2}


def die(msg):
    sys.exit(f"\n[ERROR] {msg}\n")


def warn(msg):
    print(f"[AVISO] {msg}")


def info(msg):
    print(f"[INFO]  {msg}")


def norm_element(s):
    s = s.strip()
    if not s:
        return ""
    s = s[0].upper() + s[1:].lower()
    return s if s in COV_RADII else ""


def element_from_name(name):
    """Heurística: quita dígitos y prueba 2 letras (Cl, Br...) y luego 1."""
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return ""
    two = letters[:2].capitalize()
    if two in TWO_LETTER and len(letters) >= 2:
        # 'CL1' -> Cl ;  pero 'CA' / 'CD' en ligandos suelen ser carbonos:
        if two in ("Cl", "Br"):
            return two
    return norm_element(letters[0])


def element_from_mass(m):
    best = min(MASSES, key=lambda e: abs(MASSES[e] - m))
    return best if abs(MASSES[best] - m) < 0.6 else ""


# ----------------------------------------------------------------------------
# Lectores
# ----------------------------------------------------------------------------
def read_mol2(path):
    atoms, bonds, section = [], [], None
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if s.startswith("@<TRIPOS>"):
                if s == "@<TRIPOS>MOLECULE" and atoms:
                    warn(f"{path} contiene varias moléculas; sólo se usa la primera.")
                    break
                section = s
                continue
            if not s or s.startswith("#"):
                continue
            f = s.split()
            if section == "@<TRIPOS>ATOM":
                if len(f) < 6:
                    die(f"Línea ATOM mal formada en {path}: {ln!r}")
                atype = f[5]
                el = norm_element(atype.split(".")[0]) if "." in atype else ""
                if not el:
                    el = element_from_name(f[1])
                atoms.append(dict(id=int(f[0]), name=f[1],
                                  xyz=np.array(list(map(float, f[2:5]))),
                                  type=atype, element=el,
                                  resname=f[7] if len(f) > 7 else "UNL"))
            elif section == "@<TRIPOS>BOND":
                bonds.append((int(f[1]), int(f[2])))
    if not atoms:
        die(f"No se encontraron átomos en {path} (¿falta @<TRIPOS>ATOM?).")
    ids = [a["id"] for a in atoms]
    if len(set(ids)) != len(ids):
        die(f"IDs de átomo repetidos en {path}.")
    idx = {i: k for k, i in enumerate(ids)}
    try:
        bonds = [(idx[a], idx[b]) for a, b in bonds]
    except KeyError as e:
        die(f"El BOND del mol2 referencia un átomo inexistente: {e}")
    return atoms, bonds


def read_itp_atoms(path):
    """Devuelve (moleculetype, lista de átomos del primer [ atoms ])."""
    atoms, section, molname = [], None, None
    with open(path) as fh:
        for ln in fh:
            s = ln.split(";")[0].strip()
            if not s:
                continue
            if s.startswith("["):
                new = s.strip("[] ").lower()
                if section == "atoms" and new != "atoms" and atoms:
                    section = "done"
                if section != "done":
                    section = new
                continue
            if s.startswith("#"):
                continue
            f = s.split()
            if section == "moleculetype" and molname is None:
                molname = f[0]
            elif section == "atoms":
                if len(f) < 7:
                    die(f"Línea [ atoms ] mal formada en {path}: {ln!r}")
                atoms.append(dict(nr=int(f[0]), type=f[1], resnr=int(f[2]),
                                  resname=f[3], name=f[4],
                                  charge=float(f[6]),
                                  mass=float(f[7]) if len(f) > 7 else None))
    if not atoms:
        die(f"No se encontró sección [ atoms ] en {path}.")
    return molname, atoms


def parse_pdb_atom(ln):
    name = ln[12:16].strip()
    el = norm_element(ln[76:78]) if len(ln) >= 78 else ""
    if not el:
        el = element_from_name(name)
    return dict(record=ln[:6].strip(), name=name, altloc=ln[16],
                resname=ln[17:21].strip(), chain=ln[21], resid=ln[22:26].strip(),
                icode=ln[26], xyz=np.array([float(ln[30:38]), float(ln[38:46]),
                                            float(ln[46:54])]),
                element=el)


# ----------------------------------------------------------------------------
# Emparejamiento
# ----------------------------------------------------------------------------
def bond_check(xyz, bonds, elems, tol=0.45):
    """Enlaces del mol2 que en el PDB tienen longitud imposible."""
    bad = []
    for i, j in bonds:
        d = np.linalg.norm(xyz[i] - xyz[j])
        ref = COV_RADII.get(elems[i], 0.8) + COV_RADII.get(elems[j], 0.8)
        if d > ref + tol or d < 0.5:
            bad.append((i, j, d))
    return bad


def match_by_name(mol2, bonds, lig):
    m2names = [a["name"] for a in mol2]
    pdbnames = [a["name"] for a in lig]
    if len(set(m2names)) != len(m2names):
        return None, "hay nombres repetidos en el mol2"
    if len(set(pdbnames)) != len(pdbnames):
        return None, "hay nombres repetidos en el ligando del PDB"
    if set(m2names) != set(pdbnames):
        diff = sorted(set(m2names) ^ set(pdbnames))
        return None, f"los nombres no coinciden (diferencias: {diff[:10]}...)"
    pos = {n: k for k, n in enumerate(pdbnames)}
    mapping = [pos[n] for n in m2names]  # mapping[i_mol2] = j_pdb
    for i, j in enumerate(mapping):
        if mol2[i]["element"] and lig[j]["element"] and \
           mol2[i]["element"] != lig[j]["element"]:
            return None, f"elemento distinto en {mol2[i]['name']}"
    xyz = np.array([lig[j]["xyz"] for j in mapping])
    bad = bond_check(xyz, bonds, [a["element"] for a in mol2])
    if bad:
        i, j, d = bad[0]
        return None, (f"{len(bad)} enlaces del mol2 tienen longitudes irreales en el PDB "
                      f"(p. ej. {mol2[i]['name']}-{mol2[j]['name']} = {d:.2f} Å): "
                      "los nombres coinciden pero no corresponden a los mismos átomos")
    return mapping, "ok"


def pdb_graph(lig, scale=1.15):
    g = nx.Graph()
    xyz = np.array([a["xyz"] for a in lig])
    for k, a in enumerate(lig):
        g.add_node(k, element=a["element"])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    for i in range(len(lig)):
        for j in range(i + 1, len(lig)):
            ri = COV_RADII.get(lig[i]["element"], 0.8)
            rj = COV_RADII.get(lig[j]["element"], 0.8)
            if 0.5 < d[i, j] < scale * (ri + rj) + 0.1:
                g.add_edge(i, j)
    return g


def match_by_graph(mol2, bonds, lig, scale=1.15, max_iso=20000):
    if nx is None:
        die("El modo 'graph' requiere networkx: pip install networkx")
    gm = nx.Graph()
    for k, a in enumerate(mol2):
        gm.add_node(k, element=a["element"])
    gm.add_edges_from(bonds)
    gp = pdb_graph(lig, scale)

    info(f"Grafo mol2: {gm.number_of_nodes()} átomos / {gm.number_of_edges()} enlaces; "
         f"grafo PDB (por distancias): {gp.number_of_nodes()} / {gp.number_of_edges()}")
    if gm.number_of_edges() != gp.number_of_edges():
        extra = {tuple(sorted(e)) for e in gp.edges()}
        warn("El número de enlaces difiere: la geometría del PDB puede tener "
             "contactos muy cortos o enlaces estirados. Prueba --bond-scale "
             "(p. ej. 1.10 o 1.25).")
        _ = extra
    if Counter(nx.get_node_attributes(gm, "element").values()) != \
       Counter(nx.get_node_attributes(gp, "element").values()):
        die("La composición elemental del mol2 y del ligando en el PDB es distinta:\n"
            f"  mol2: {dict(Counter(a['element'] for a in mol2))}\n"
            f"  PDB : {dict(Counter(a['element'] for a in lig))}\n"
            "  ¿Faltan hidrógenos en el PDB o están protonados distinto?")

    nm = iso.categorical_node_match("element", "")
    GM = iso.GraphMatcher(gm, gp, node_match=nm)
    # Criterio: 1) menor RMSD (Kabsch, redondeado a 0.01 Å) entre la geometría
    # del mol2 y la pose -> resuelve permutaciones de átomos equivalentes por
    # simetría (H de metilos, O de carboxilatos...) de la forma más fiel;
    # 2) desempate: más nombres conservados. (Si se llegó aquí, los nombres
    # ya demostraron no ser fiables, por eso la geometría va primero.)
    best, best_key, n = None, None, 0
    names_p = [a["name"] for a in lig]
    ref_xyz = np.array([a["xyz"] for a in mol2])
    lig_xyz = np.array([a["xyz"] for a in lig])
    for m in GM.isomorphisms_iter():  # m: nodo_mol2 -> nodo_pdb
        n += 1
        score = sum(mol2[i]["name"] == names_p[j] for i, j in m.items())
        rmsd = kabsch_rmsd(ref_xyz, lig_xyz[[m[i] for i in range(len(mol2))]])
        key = (-round(rmsd, 2), score)
        if best_key is None or key > best_key:
            best, best_key = m, key
        if n >= max_iso or (score == len(mol2) and rmsd < 0.05):
            break
    best_score = best_key[1] if best_key else 0
    if best is None:
        die("No se encontró isomorfismo entre el mol2 y el ligando del PDB. "
            "Causas típicas: distinta protonación/tautómero, átomos faltantes, "
            "o conectividad mal inferida (ajusta --bond-scale).")
    info(f"Isomorfismos evaluados: {n}{' (límite alcanzado)' if n >= max_iso else ''}; "
         f"nombres conservados en el mejor mapeo: {best_score}/{len(mol2)}")
    return [best[i] for i in range(len(mol2))]


def kabsch_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a0 @ r - b0) ** 2).sum(1).mean()))


# ----------------------------------------------------------------------------
# Escritores
# ----------------------------------------------------------------------------
def fmt_name(name, element):
    """Columnas 13-16 del PDB: nombres de 4 caracteres o elemento de 2 letras
    empiezan en la col. 13; el resto, en la 14."""
    if len(name) >= 4 or len(element) == 2:
        return f"{name:<4s}"[:4]
    return f" {name:<3s}"


def pdb_line(record, serial, name, resname, chain, resid, xyz, element, icode=" "):
    return (f"{record:<6s}{serial % 100000:5d} {fmt_name(name, element)} "
            f"{resname[:4]:<4s}{chain}{resid:>4s}{icode}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {element:>2s}\n")


def write_gro(path, names, resname, xyz_A, title):
    with open(path, "w") as fh:
        fh.write(f"{title}\n{len(names)}\n")
        for k, (n, x) in enumerate(zip(names, xyz_A), 1):
            x = x / 10.0
            fh.write(f"{1:5d}{resname[:5]:<5s}{n[:5]:>5s}{k % 100000:5d}"
                     f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}\n")
        span = xyz_A.max(0) / 10.0 - xyz_A.min(0) / 10.0 + 2.0
        fh.write(f"{span[0]:10.5f}{span[1]:10.5f}{span[2]:10.5f}\n")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Reordena el ligando de un PDB de complejo al orden del mol2/itp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--complex", required=True, help="PDB con proteína + ligando posicionado")
    ap.add_argument("--mol2", required=True, help="mol2 del ligando usado para generar el itp")
    ap.add_argument("--itp", help="itp del ligando (recomendado: se valida contra él)")
    ap.add_argument("--resname", help="resname del ligando en el PDB del complejo "
                    "(default: el del mol2)")
    ap.add_argument("--chain", help="cadena del ligando, si hay ambigüedad")
    ap.add_argument("--resid", help="número de residuo del ligando, si hay varias copias")
    ap.add_argument("--match", choices=["auto", "name", "graph"], default="auto")
    ap.add_argument("--bond-scale", type=float, default=1.15,
                    help="factor sobre la suma de radios covalentes para inferir enlaces")
    ap.add_argument("--out", required=True, help="PDB de salida del complejo")
    ap.add_argument("--out-lig", help="ligando solo, en orden del mol2 (.pdb o .gro)")
    ap.add_argument("--hetatm", action="store_true",
                    help="escribir el ligando como HETATM (default: conserva el registro)")
    args = ap.parse_args()

    print(f"reorder_ligand_by_mol2.py v{__version__}")

    # ---- mol2 ----
    mol2, bonds = read_mol2(args.mol2)
    info(f"mol2: {len(mol2)} átomos, {len(bonds)} enlaces, resname '{mol2[0]['resname']}'")
    if not bonds:
        warn("El mol2 no tiene sección @<TRIPOS>BOND: el modo 'graph' no funcionará.")

    # ---- itp (opcional) ----
    out_names = [a["name"] for a in mol2]
    lig_resname_out = mol2[0]["resname"]
    if args.itp:
        molname, itp = read_itp_atoms(args.itp)
        info(f"itp: moleculetype '{molname}', {len(itp)} átomos, "
             f"carga neta {sum(a['charge'] for a in itp):+.4f}")
        if len(itp) != len(mol2):
            die(f"El itp tiene {len(itp)} átomos y el mol2 {len(mol2)}: "
                "no corresponden a la misma molécula (¿otro mol2?).")
        if [a["nr"] for a in itp] != list(range(1, len(itp) + 1)):
            warn("La numeración del [ atoms ] del itp no es consecutiva desde 1.")
        # elemento por masa: fuente más fiable (tipos GAFF en minúsculas confunden)
        for a_m, a_i in zip(mol2, itp):
            if a_i["mass"]:
                el = element_from_mass(a_i["mass"])
                if el:
                    if a_m["element"] and a_m["element"] != el:
                        warn(f"Elemento de {a_m['name']}: mol2 sugiere "
                             f"{a_m['element']}, masa del itp indica {el}; se usa {el}.")
                    a_m["element"] = el
        mism = [(k + 1, m["name"], i["name"]) for k, (m, i) in enumerate(zip(mol2, itp))
                if m["name"] != i["name"]]
        if mism:
            warn(f"{len(mism)} nombres difieren entre mol2 e itp en la misma posición "
                 f"(p. ej. #{mism[0][0]}: mol2 {mism[0][1]} / itp {mism[0][2]}). "
                 "Se asume el mismo orden (lo que importa para grompp) y se "
                 "escriben los nombres del itp.")
        out_names = [a["name"] for a in itp]
        resn = {a["resname"] for a in itp}
        if len(resn) > 1:
            warn(f"El itp tiene varios resnames {resn}; se usa el primero.")
        lig_resname_out = itp[0]["resname"]

    missing_el = [a["name"] for a in mol2 if not a["element"]]
    if missing_el:
        die(f"No se pudo deducir el elemento de {missing_el[:10]} en el mol2. "
            "Proporciona --itp (se deduce por masa).")

    # ---- PDB del complejo ----
    with open(args.complex) as fh:
        lines = fh.readlines()
    resname = args.resname or mol2[0]["resname"]
    lig_idx, lig = [], []
    other_resnames = Counter()
    for k, ln in enumerate(lines):
        if ln.startswith(("ATOM  ", "HETATM")):
            a = parse_pdb_atom(ln)
            if a["resname"] == resname and \
               (args.chain is None or a["chain"] == args.chain) and \
               (args.resid is None or a["resid"] == str(args.resid)):
                lig_idx.append(k)
                lig.append(a)
            elif ln.startswith("HETATM"):
                other_resnames[a["resname"]] += 1
    if any(l.startswith("ENDMDL") for l in lines):
        n_models = sum(l.startswith("MODEL") for l in lines)
        if n_models > 1:
            die(f"El PDB tiene {n_models} modelos; extrae uno antes (p. ej. con "
                "gmx trjconv -dump o pdb_selmodel).")
    if not lig:
        die(f"No hay átomos con resname '{resname}' en {args.complex}. "
            f"HETATM presentes: {dict(other_resnames)}. Usa --resname.")

    # altlocs
    alts = {a["altloc"] for a in lig if a["altloc"] not in (" ", "")}
    if alts:
        keep = sorted(alts)[0]
        warn(f"Ligando con altlocs {alts}; se conserva '{keep}'.")
        sel = [k for k, a in enumerate(lig) if a["altloc"] in (" ", keep)]
        drop = set(lig_idx) - {lig_idx[k] for k in sel}
        lig = [lig[k] for k in sel]
        lig_idx = [lig_idx[k] for k in sel]
    else:
        drop = set()

    copies = {(a["chain"], a["resid"]) for a in lig}
    if len(copies) > 1:
        die(f"Hay {len(copies)} residuos '{resname}' en el PDB {sorted(copies)}. "
            "Selecciona uno con --chain/--resid (y repite para cada copia).")
    if len(lig) != len(mol2):
        nh_p = sum(a["element"] == "H" for a in lig)
        nh_m = sum(a["element"] == "H" for a in mol2)
        die(f"El ligando del PDB tiene {len(lig)} átomos ({nh_p} H) y el mol2 "
            f"{len(mol2)} ({nh_m} H). Si difieren los H, el visor los quitó/añadió: "
            "vuelve a exportar conservando los H del mol2.")
    if len(set(lig_idx)) and lig_idx != list(range(lig_idx[0], lig_idx[0] + len(lig_idx))):
        warn("Los átomos del ligando no son contiguos en el PDB; se escribirán "
             "juntos en la posición de su primer átomo.")
    info(f"Ligando en PDB: resname {resname}, cadena '{lig[0]['chain']}', "
         f"resid {lig[0]['resid']}, {len(lig)} átomos")

    # ---- emparejamiento ----
    mapping = None
    if args.match in ("auto", "name"):
        mapping, why = match_by_name(mol2, bonds, lig)
        if mapping is not None:
            info("Emparejamiento por nombre: OK")
        elif args.match == "name":
            die(f"Emparejamiento por nombre falló: {why}")
        else:
            info(f"Emparejamiento por nombre no aplicable ({why}); se usa el grafo.")
    if mapping is None:
        mapping = match_by_graph(mol2, bonds, lig, args.bond_scale)

    if len(set(mapping)) != len(mapping):
        die("Mapeo no biyectivo (bug o entrada inconsistente).")

    new_xyz = np.array([lig[j]["xyz"] for j in mapping])

    # ---- verificaciones del resultado ----
    for i, j in enumerate(mapping):
        if mol2[i]["element"] != lig[j]["element"]:
            die(f"Elemento inconsistente tras el mapeo: {mol2[i]['name']}")
    bad = bond_check(new_xyz, bonds, [a["element"] for a in mol2])
    if bad:
        die(f"{len(bad)} enlaces del mol2 quedan con longitud irreal tras reordenar; "
            "el mapeo no es fiable.")
    rmsd = kabsch_rmsd(np.array([a["xyz"] for a in mol2]), new_xyz)
    info(f"RMSD (tras superponer) mol2 vs. pose del complejo: {rmsd:.3f} Å "
         f"{'(misma conformación)' if rmsd < 0.1 else '(conformación distinta: normal si viene de docking)'}")
    renamed = sum(lig[j]["name"] != mol2[i]["name"] for i, j in enumerate(mapping))
    if renamed:
        info(f"{renamed} átomos cambian de nombre para coincidir con el mol2/itp.")

    # ---- escritura del complejo ----
    ref = lig[0]
    rec = "HETATM" if args.hetatm else ref["record"]
    lig_set = set(lig_idx) | drop
    first = min(lig_idx)
    out, serial, dropped_conect = [], 0, 0
    for k, ln in enumerate(lines):
        if ln.startswith("CONECT"):
            dropped_conect += 1
            continue
        if k in lig_set:
            if k == first:
                for n, x, a in zip(out_names, new_xyz, mol2):
                    serial += 1
                    out.append(pdb_line(rec, serial, n, lig_resname_out, ref["chain"],
                                        ref["resid"], x, a["element"].upper(), ref["icode"]))
            continue
        if ln.startswith(("ATOM  ", "HETATM")):
            serial += 1
            out.append(ln[:6] + f"{serial % 100000:5d}" + ln[11:])
        elif ln.startswith("TER"):
            out.append("TER\n")  # sin número: evita desfasar la numeración
        else:
            out.append(ln)
    if dropped_conect:
        warn(f"Se eliminaron {dropped_conect} registros CONECT (la numeración cambió; "
             "GROMACS no los usa).")
    with open(args.out, "w") as fh:
        fh.write(f"REMARK   Ligando {lig_resname_out} reordenado segun {args.mol2}"
                 f"{' / ' + args.itp if args.itp else ''} con reorder_ligand_by_mol2.py\n")
        fh.writelines(out)
    info(f"Complejo escrito: {args.out}")

    # ---- re-lectura de control ----
    with open(args.out) as fh:
        chk = [parse_pdb_atom(l) for l in fh if l.startswith(("ATOM  ", "HETATM"))
               and l[17:21].strip() == lig_resname_out[:4]]
    if [a["name"] for a in chk] != [n[:4] for n in out_names]:
        die("Control final falló: el orden de nombres en la salida no coincide con el mol2/itp.")
    info("Control final: orden del ligando en la salida == orden del mol2/itp  ✔")

    # ---- ligando aparte ----
    if args.out_lig:
        if args.out_lig.lower().endswith(".gro"):
            write_gro(args.out_lig, out_names, lig_resname_out, new_xyz,
                      f"{lig_resname_out} in mol2/itp order")
        else:
            with open(args.out_lig, "w") as fh:
                for k, (n, x, a) in enumerate(zip(out_names, new_xyz, mol2), 1):
                    fh.write(pdb_line("HETATM", k, n, lig_resname_out, ref["chain"],
                                      ref["resid"], x, a["element"].upper()))
                fh.write("END\n")
        info(f"Ligando escrito: {args.out_lig}")


if __name__ == "__main__":
    main()
