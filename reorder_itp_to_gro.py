#!/usr/bin/env python3
"""
reorder_itp_to_gro.py
=====================
Reordena los átomos de un [ moleculetype ] de un archivo .itp para que sigan
EXACTAMENTE el orden en que aparecen en un archivo .gro (o .pdb), renumerando
[ atoms ] y remapeando los índices de TODAS las secciones que los usan
(bonds, pairs, angles, dihedrals, exclusions, virtual_sites, restraints...).

El problema que resuelve: antechamber/acpype y CHARMM-GUI parten de la misma
molécula pero pueden escribir los átomos en distinto orden. GROMACS ignora los
nombres y asocia topología y coordenadas POR POSICIÓN, así que un desfase de
orden produce una molécula con la conectividad equivocada aunque grompp solo
emita warnings.

Cómo decide la correspondencia (en este orden):

  1. POR NOMBRE. Si los nombres del itp y del .gro son los mismos conjuntos y
     no se repiten, se empareja por nombre. Se VALIDA midiendo, sobre las
     coordenadas del .gro, todos los enlaces de [ bonds ] del itp. Si algún
     enlace sale absurdo, el emparejamiento por nombre se descarta (nombres
     iguales pueden estar puestos sobre átomos físicamente distintos).

  2. POR GRAFO MOLECULAR. Se construye el grafo del itp con [ bonds ] y el del
     .gro por distancias interatómicas, y se busca un isomorfismo con
     refinamiento tipo Weisfeiler-Lehman + backtracking. Si la molécula tiene
     simetría topológica se buscan varias soluciones, se escoge la que más
     coincide con los nombres y se AVISA si las alternativas asignarían tipos
     o cargas distintas al mismo átomo.

  3. Si nada funciona, no escribe nada y explica por qué.

Verificaciones internas incluidas:
  - numeración de [ atoms ] consecutiva 1..N y sin huecos
  - mismo número de átomos en itp y en el residuo del .gro
  - índices de todas las secciones dentro de rango, antes y después
  - elementos deducidos del .gro consistentes con las masas del itp
  - longitudes de enlace del itp medidas sobre el .gro (con imagen mínima,
    cajas rectangulares y triclínicas)
  - carga total y número de líneas por sección conservados tras el reordenado
  - todas las copias del residuo en el .gro tienen el mismo orden interno

Requisitos: Python >= 3.8, numpy
Uso típico:
  python reorder_itp_to_gro.py -i LIG.itp -g step5_input.gro -r LIG \
         -o LIG_reordenado.itp
"""
import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Tablas químicas
# ---------------------------------------------------------------------------
MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
          "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.90,
          "NA": 22.990, "MG": 24.305, "K": 39.098, "CA": 40.078,
          "ZN": 65.38, "FE": 55.845, "SE": 78.971, "SI": 28.085, "B": 10.81}
# radios covalentes (nm)
RCOV = {"H": 0.031, "C": 0.076, "N": 0.071, "O": 0.066, "F": 0.057,
        "P": 0.107, "S": 0.105, "CL": 0.102, "BR": 0.120, "I": 0.139,
        "SE": 0.120, "SI": 0.111, "B": 0.084, "ZN": 0.122, "FE": 0.132,
        "NA": 0.166, "MG": 0.141, "K": 0.203, "CA": 0.176}
TWO_LETTER = {"CL", "BR", "NA", "MG", "ZN", "FE", "SE", "SI", "CA", "LI", "CU", "MN"}

# Secciones de un [ moleculetype ] y cuántas columnas iniciales son índices
# de átomo. "all" = todas las columnas; "vsn" = col 1 e índices desde la col 3.
SECT_IDX = {
    "bonds": 2, "pairs": 2, "pairs_nb": 2, "angles": 3, "dihedrals": 4,
    "constraints": 2, "settles": 1, "exclusions": "all",
    "position_restraints": 1, "distance_restraints": 2,
    "dihedral_restraints": 4, "angle_restraints": 4, "angle_restraints_z": 2,
    "orientation_restraints": 2, "cmap": 5,
    "virtual_sites1": 2, "virtual_sites2": 3, "virtual_sites3": 4,
    "virtual_sites4": 5, "virtual_sitesn": "vsn",
}


def die(msg):
    sys.exit(f"[ERROR] {msg}")


def element_from_mass(m):
    if m is None:
        return None
    best, err = None, 1e9
    for el, mm in MASSES.items():
        e = abs(mm - m)
        if e < err:
            best, err = el, e
    return best if err < 0.6 else None


def element_from_name(name, allowed=None):
    """Elemento a partir del nombre del átomo. `allowed` es el Counter de
    elementos que el itp dice que debe haber; se usa para desambiguar casos
    como CL1 (¿cloro o carbono?)."""
    s = re.sub(r"[^A-Za-z]", "", name).upper()
    if not s:
        return None
    cands = []
    if len(s) >= 2 and s[:2] in TWO_LETTER:
        cands.append(s[:2])
    if s[0] in MASSES:
        cands.append(s[0])
    if not cands:
        return None
    if allowed:
        for c in cands:
            if allowed.get(c, 0) > 0:
                return c
    return cands[0]


# ---------------------------------------------------------------------------
# Lectura del .itp conservando el archivo completo
# ---------------------------------------------------------------------------
class ItpFile:
    """Guarda el archivo como lista de registros para poder reescribirlo
    entero cambiando solo las secciones del moleculetype elegido."""

    def __init__(self, path):
        self.path = path
        self.records = []      # (tipo, seccion, molname, texto)
        self.mols = []         # dicts con name, atoms
        cur, section = None, None
        with open(path) as fh:
            raw_lines = fh.read().splitlines()

        for lineno, raw in enumerate(raw_lines, 1):
            stripped = raw.strip()
            code = raw.split(";", 1)[0].strip()

            if stripped.startswith("#"):                       # preprocesador
                self.records.append(["raw", section, cur["name"] if cur else None, raw])
                continue
            m = re.match(r"^\[\s*([A-Za-z_0-9]+)\s*\]", code)
            if m:
                section = m.group(1).lower()
                if section == "moleculetype":
                    cur = {"name": None, "atoms": [], "bonds": []}
                    self.mols.append(cur)
                self.records.append(["raw", section, cur["name"] if cur else None, raw])
                continue
            if not code:                                       # vacía o comentario
                self.records.append(["raw", section, cur["name"] if cur else None, raw])
                continue

            f = code.split()
            comment = raw.split(";", 1)[1].rstrip() if ";" in raw else None

            if cur is not None and section == "moleculetype" and cur["name"] is None:
                cur["name"] = f[0]
                self.records.append(["raw", section, cur["name"], raw])
                continue

            molname = cur["name"] if cur else None

            if cur is not None and section == "atoms":
                if len(f) < 5:
                    die(f"{path}:{lineno}: línea de [ atoms ] incompleta: {raw.strip()}")
                try:
                    atom = {"nr": int(f[0]), "type": f[1], "resnr": f[2],
                            "resname": f[3], "name": f[4],
                            "cgnr": f[5] if len(f) > 5 else f[0],
                            "charge": float(f[6]) if len(f) > 6 else 0.0,
                            "mass": float(f[7]) if len(f) > 7 else None,
                            "extra": f[8:], "comment": comment}
                except ValueError:
                    die(f"{path}:{lineno}: no se pudieron leer los números de "
                        f"[ atoms ]: {raw.strip()}")
                cur["atoms"].append(atom)
                self.records.append(["atom", section, molname, raw])
                continue

            if cur is not None and section in SECT_IDX:
                spec = SECT_IDX[section]
                try:
                    if spec == "all":
                        idx = [int(x) for x in f]
                        rest = []
                    elif spec == "vsn":
                        idx = [int(f[0])] + [int(x) for x in f[2:]]
                        rest = [f[1]]
                    else:
                        idx = [int(x) for x in f[:spec]]
                        rest = f[spec:]
                except ValueError:
                    die(f"{path}:{lineno}: índices no numéricos en "
                        f"[ {section} ]: {raw.strip()}")
                if section == "bonds":
                    cur["bonds"].append((idx[0], idx[1]))
                self.records.append(["idx", section, molname,
                                     {"idx": idx, "rest": rest, "comment": comment,
                                      "raw": raw, "lineno": lineno}])
                continue

            self.records.append(["raw", section, molname, raw])

        if not self.mols:
            die(f"{path} no contiene ningún [ moleculetype ].")

    def get_mol(self, molname=None):
        if molname:
            sel = [m for m in self.mols if m["name"] == molname]
            if not sel:
                die(f"No existe la molécula '{molname}' en {self.path}. "
                    f"Disponibles: {[m['name'] for m in self.mols]}")
            return sel[0]
        if len(self.mols) == 1:
            return self.mols[0]
        die(f"{self.path} tiene varias moléculas "
            f"{[m['name'] for m in self.mols]}; especifica cuál con --molname.")


# ---------------------------------------------------------------------------
# Lectura del .gro / .pdb
# ---------------------------------------------------------------------------
def parse_gro(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    if len(lines) < 3:
        die(f"{path} está vacío o incompleto.")
    try:
        natoms = int(lines[1].strip())
    except ValueError:
        die(f"La línea 2 de {path} debería ser el número de átomos.")
    if len(lines) < natoms + 3:
        die(f"{path} declara {natoms} átomos pero solo tiene "
            f"{len(lines) - 3} líneas de átomos (archivo truncado).")
    atom_lines = lines[2:2 + natoms]
    box = [float(x) for x in lines[2 + natoms].split()]
    if len(box) not in (3, 9):
        die(f"Línea de caja inválida en {path}: '{lines[2 + natoms]}'")

    dots = [k for k, c in enumerate(atom_lines[0]) if c == "." and k >= 20]
    if len(dots) < 3:
        die(f"No se pudo interpretar el formato de {path} (línea de átomo 1).")
    w = dots[1] - dots[0]

    atoms = []
    for k, l in enumerate(atom_lines):
        try:
            xyz = [float(l[20 + d * w: 20 + (d + 1) * w]) for d in range(3)]
        except ValueError:
            die(f"Coordenadas ilegibles en la línea {k + 3} de {path}.")
        atoms.append({"resid": l[0:5].strip(), "resname": l[5:10].strip(),
                      "name": l[10:15].strip(), "xyz": xyz})
    return atoms, box


def parse_pdb(path):
    atoms = []
    box = None
    with open(path) as fh:
        for l in fh:
            if l.startswith("CRYST1"):
                try:
                    box = [float(l[6:15]) / 10, float(l[15:24]) / 10, float(l[24:33]) / 10]
                except ValueError:
                    box = None
            elif l.startswith(("ATOM  ", "HETATM")):
                try:
                    xyz = [float(l[30:38]) / 10, float(l[38:46]) / 10, float(l[46:54]) / 10]
                except ValueError:
                    die(f"Coordenadas ilegibles en {path}: {l.rstrip()}")
                atoms.append({"resid": l[22:27].strip(), "resname": l[17:21].strip(),
                              "name": l[12:16].strip(), "xyz": xyz})
    if not atoms:
        die(f"{path} no contiene líneas ATOM/HETATM.")
    if box is None:
        box = [0.0, 0.0, 0.0]        # sin PBC
    return atoms, box


def box_vectors(box):
    if len(box) == 3:
        return np.diag(box)
    v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = box
    return np.array([[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]])


SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def pair_distances(coords, pairs, bvec):
    """Distancias con convención de imagen mínima."""
    if len(pairs) == 0:
        return np.array([])
    i, j = np.array(pairs).T
    d = coords[j] - coords[i]
    shifts = SHIFTS @ bvec
    return np.linalg.norm(d[:, None, :] + shifts[None], axis=2).min(axis=1)


def all_distances(coords, bvec):
    n = len(coords)
    d = coords[:, None, :] - coords[None, :, :]
    shifts = SHIFTS @ bvec
    out = np.linalg.norm(d[:, :, None, :] + shifts[None, None], axis=3).min(axis=2)
    out[np.arange(n), np.arange(n)] = 1e9
    return out


# ---------------------------------------------------------------------------
# Grafos e isomorfismo
# ---------------------------------------------------------------------------
def _adj_from_cut(d, cut):
    n = d.shape[0]
    adj = [set() for _ in range(n)]
    ii, jj = np.where(d < cut)
    for a, b in zip(ii, jj):
        if a < b:
            adj[a].add(int(b))
            adj[b].add(int(a))
    return adj


def infer_bonds(coords, elements, bvec, target_nb, target_deg):
    """Conectividad del .gro por distancias. Primero con radios covalentes a
    partir de los elementos deducidos de los nombres; si eso no reproduce el
    número de enlaces y la secuencia de grados del itp, se barren tolerancias y
    cortes uniformes hasta encontrar una conectividad compatible. Así el
    resultado no depende de que los nombres del .gro sean correctos."""
    d = all_distances(coords, bvec)
    r = np.array([RCOV.get(e or "C", 0.085) for e in elements])
    base = r[:, None] + r[None, :]
    tried = []

    def ok(adj):
        nb = sum(len(s) for s in adj) // 2
        deg = sorted(len(s) for s in adj)
        return nb == target_nb and deg == target_deg

    for tol in (1.25, 1.20, 1.30, 1.15, 1.35, 1.10, 1.40):
        adj = _adj_from_cut(d, tol * base)
        tried.append((f"radios covalentes x{tol:.2f}", adj))
        if ok(adj):
            return adj, f"radios covalentes x{tol:.2f}"
    for cut in np.arange(0.150, 0.2205, 0.002):
        adj = _adj_from_cut(d, np.full_like(base, cut))
        if ok(adj):
            return adj, f"corte uniforme {cut:.3f} nm"
    return tried[0][1], "radios covalentes x1.25 (sin coincidencia exacta)"


def wl_labels(adjs, inits, rounds=4):
    """Refinamiento Weisfeiler-Lehman simultáneo sobre varios grafos, con la
    misma tabla de compresión, para que las etiquetas sean comparables."""
    labs = [list(i) for i in inits]
    for _ in range(rounds):
        table, new_labs = {}, []
        for adj, lab in zip(adjs, labs):
            nl = []
            for v in range(len(adj)):
                key = (lab[v], tuple(sorted(lab[u] for u in adj[v])))
                nl.append(table.setdefault(key, len(table)))
            new_labs.append(nl)
        if new_labs == labs:
            break
        labs = new_labs
    return labs


def find_isomorphisms(adjA, labA, adjB, labB, max_maps=64):
    """Mapea cada nodo de A (itp) a uno de B (gro). Devuelve lista de tuplas
    map[i] = j."""
    n = len(adjA)
    if len(adjB) != n:
        return []
    cand = []
    byl = defaultdict(list)
    for j, l in enumerate(labB):
        byl[l].append(j)
    for i in range(n):
        c = byl.get(labA[i], [])
        if not c:
            return []
        cand.append(c)

    order = sorted(range(n), key=lambda i: (len(cand[i]), -len(adjA[i])))
    # reordenar para que cada nodo se conecte con los ya colocados
    seq, placed = [], set()
    remaining = list(order)
    while remaining:
        pick = next((i for i in remaining if adjA[i] & placed), remaining[0])
        remaining.remove(pick)
        seq.append(pick)
        placed.add(pick)

    mapping = [-1] * n
    used = [False] * n
    out = []

    def bt(k):
        if len(out) >= max_maps:
            return
        if k == n:
            out.append(tuple(mapping))
            return
        i = seq[k]
        for j in cand[i]:
            if used[j]:
                continue
            ok = True
            for u in adjA[i]:
                if mapping[u] != -1 and mapping[u] not in adjB[j]:
                    ok = False
                    break
            if ok:
                # grado compatible y vecinos ya mapeados de j deben estar en adjA[i]
                if len(adjB[j]) != len(adjA[i]):
                    continue
                for v in adjB[j]:
                    iu = next((u for u in range(n) if mapping[u] == v), None)
                    if iu is not None and iu not in adjA[i]:
                        ok = False
                        break
            if ok:
                mapping[i], used[j] = j, True
                bt(k + 1)
                mapping[i], used[j] = -1, False

    bt(0)
    return out


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
def fmt_atom(a, nr, cgnr, qtot):
    com = f"   ; qtot {qtot:+.3f}"
    mass = f"{a['mass']:11.5f}" if a["mass"] is not None else ""
    return (f"{nr:6d} {a['type']:>10s} {a['resnr']:>6s} {a['resname']:>6s} "
            f"{a['name']:>6s} {cgnr:6d} {a['charge']:12.6f}{mass}{com}")


def fmt_idx(idx, rest, comment):
    s = "".join(f"{i:6d}" for i in idx)
    if rest:
        s += "  " + " ".join(rest)
    if comment:
        s += f"   ;{comment}"
    return s


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--itp", required=True, help="archivo .itp a reordenar")
    ap.add_argument("-g", "--gro", required=True, help=".gro o .pdb que define el orden")
    ap.add_argument("-r", "--resname", help="nombre del residuo en el .gro "
                                            "(por defecto el del itp)")
    ap.add_argument("-m", "--molname", help="[ moleculetype ] a reordenar si hay varios")
    ap.add_argument("-o", "--out", help="itp de salida (si se omite, solo diagnostica)")
    ap.add_argument("--names", choices=["gro", "itp"], default="gro",
                    help="nombres de átomo del itp de salida: los del .gro "
                         "(por defecto, elimina los warnings de grompp) o los del itp")
    ap.add_argument("--keep-comments", action="store_true",
                    help="conservar los comentarios originales en vez de "
                         "regenerarlos con los nombres nuevos")
    ap.add_argument("--no-sort", action="store_true",
                    help="no ordenar las líneas dentro de cada sección")
    ap.add_argument("--lo", type=float, default=0.07, help="enlace mínimo en nm (0.07)")
    ap.add_argument("--hi", type=float, default=0.25, help="enlace máximo en nm (0.25)")
    ap.add_argument("--accept-ambiguous", action="store_true",
                    help="continuar aunque el grafo tenga simetrías que hagan "
                         "ambigua la asignación de cargas")
    args = ap.parse_args()

    # ---------------- itp ----------------
    itp = ItpFile(args.itp)
    mol = itp.get_mol(args.molname)
    atoms = mol["atoms"]
    nat = len(atoms)
    if nat == 0:
        die(f"La molécula '{mol['name']}' no tiene sección [ atoms ].")
    if [a["nr"] for a in atoms] != list(range(1, nat + 1)):
        die("La numeración de [ atoms ] no es consecutiva 1..N; "
            "corrige el itp antes de reordenarlo.")
    for rec in itp.records:
        if rec[0] == "idx" and rec[2] == mol["name"]:
            for v in rec[3]["idx"]:
                if not 1 <= v <= nat:
                    die(f"{args.itp}:{rec[3]['lineno']}: índice {v} fuera de "
                        f"rango 1..{nat} en [ {rec[1]} ].")
    itp_names = [a["name"] for a in atoms]
    itp_el = [element_from_mass(a["mass"]) for a in atoms]
    if any(e is None for e in itp_el):
        sin = [atoms[k]["name"] for k, e in enumerate(itp_el) if e is None]
        print(f"[AVISO] No se pudo deducir el elemento por masa de: {sin[:10]}"
              f"{'...' if len(sin) > 10 else ''}. Se usará solo H/pesado.")
    print(f"[INFO] itp: molécula '{mol['name']}', {nat} átomos, "
          f"{len(mol['bonds'])} enlaces, carga total "
          f"{sum(a['charge'] for a in atoms):+.4f}")
    if not mol["bonds"]:
        die("El [ moleculetype ] no tiene [ bonds ]; sin conectividad no se "
            "puede validar ni deducir la correspondencia.")

    # ---------------- gro ----------------
    if args.gro.lower().endswith(".pdb"):
        gatoms, box = parse_pdb(args.gro)
    else:
        gatoms, box = parse_gro(args.gro)
    bvec = box_vectors(box)
    resname = args.resname or atoms[0]["resname"]

    blocks, cur, prev = [], [], None
    for k, a in enumerate(gatoms):
        key = (a["resid"], a["resname"])
        if a["resname"] == resname:
            if cur and key != prev:
                blocks.append(cur)
                cur = []
            cur.append(k)
        elif cur:
            blocks.append(cur)
            cur = []
        prev = key
    if cur:
        blocks.append(cur)
    if not blocks:
        die(f"No hay residuos '{resname}' en {args.gro}. "
            f"Residuos presentes: {sorted({a['resname'] for a in gatoms})}")
    print(f"[INFO] gro: {len(blocks)} copia(s) de '{resname}'")

    for b, idx in enumerate(blocks, 1):
        if len(idx) != nat:
            die(f"La copia {b} de '{resname}' en el .gro tiene {len(idx)} átomos "
                f"y el itp {nat}. No es la misma molécula (¿protonación, "
                f"hidrógenos o residuo equivocado?).")
    ref = blocks[0]
    gro_names = [gatoms[k]["name"] for k in ref]
    for b, idx in enumerate(blocks[1:], 2):
        other = [gatoms[k]["name"] for k in idx]
        if other != gro_names:
            die(f"La copia {b} de '{resname}' tiene los átomos en otro orden que "
                f"la copia 1. Un solo [ moleculetype ] no puede describir ambas.")
    coords = np.array([gatoms[k]["xyz"] for k in ref])

    # elementos del gro, validados contra las masas del itp
    want = Counter(e for e in itp_el if e)
    gro_el, pool = [], Counter(want)
    for n_ in gro_names:
        e = element_from_name(n_, pool)
        gro_el.append(e)
        if e:
            pool[e] -= 1
    use_elements = Counter(e for e in gro_el if e) == want and all(gro_el)
    if use_elements:
        print("[INFO] Elementos deducidos del .gro coinciden con las masas del itp.")
    else:
        print("[AVISO] Los elementos deducidos de los nombres del .gro no "
              "coinciden con las masas del itp; el emparejamiento por grafo "
              "distinguirá solo H frente a átomo pesado.")

    itp_bonds0 = [(i - 1, j - 1) for i, j in mol["bonds"]]

    def validate(mapping):
        """mapping[i] = posición en el .gro del átomo i del itp."""
        pairs = [(mapping[i], mapping[j]) for i, j in itp_bonds0]
        d = pair_distances(coords, pairs, bvec)
        bad = [(itp_names[i], itp_names[j], float(dd))
               for (i, j), dd in zip(itp_bonds0, d) if dd < args.lo or dd > args.hi]
        hmis = sum(1 for i in range(nat)
                   if itp_el[i] and ((itp_el[i] == "H") !=
                                     (re.sub(r"[^A-Za-z]", "", gro_names[mapping[i]])[:1].upper() == "H")))
        return {"bad": bad, "hmis": hmis, "dmin": float(d.min()), "dmax": float(d.max()),
                "ok": not bad and hmis == 0, "geom_ok": not bad}

    # ---------------- 1) emparejamiento por nombre ----------------
    mapping, how, ambiguous = None, None, []
    if (Counter(itp_names) == Counter(gro_names)
            and len(set(itp_names)) == nat and len(set(gro_names)) == nat):
        pos = {n_: k for k, n_ in enumerate(gro_names)}
        cand_map = [pos[n_] for n_ in itp_names]
        v = validate(cand_map)
        print(f"[INFO] Nombres idénticos en ambos archivos. Validación geométrica: "
              f"{len(v['bad'])} enlaces anómalos, {v['hmis']} H inconsistentes, "
              f"rango {v['dmin']:.3f}-{v['dmax']:.3f} nm")
        if v["ok"]:
            mapping, how = cand_map, "nombre"
        else:
            print("[AVISO] El emparejamiento por nombre da una geometría "
                  "imposible: los mismos nombres están sobre átomos distintos. "
                  "Se pasa al emparejamiento por grafo.")
            for a, b, dd in v["bad"][:8]:
                print(f"        {a}-{b}: {dd:.3f} nm")
    else:
        print("[INFO] Los conjuntos de nombres difieren entre itp y .gro; "
              "se empareja por grafo molecular.")

    # ---------------- 2) emparejamiento por grafo ----------------
    if mapping is None:
        adjA = [set() for _ in range(nat)]
        for i, j in itp_bonds0:
            adjA[i].add(j)
            adjA[j].add(i)
        el_for_bonds = gro_el if use_elements else [
            ("H" if re.sub(r"[^A-Za-z]", "", n_)[:1].upper() == "H" else "C")
            for n_ in gro_names]
        nb_a = sum(len(s) for s in adjA) // 2
        deg_a = sorted(len(s) for s in adjA)
        adjB, crit = infer_bonds(coords, el_for_bonds, bvec, nb_a, deg_a)
        nb_b = sum(len(s) for s in adjB) // 2
        print(f"[INFO] Enlaces: {nb_a} en el itp, {nb_b} detectados en el .gro "
              f"({crit})")
        if nb_a != nb_b or sorted(len(s) for s in adjB) != deg_a:
            die(f"No se encontró ningún criterio de distancia que reproduzca la "
                f"conectividad del itp ({nb_a} enlaces, grados {Counter(deg_a)}); "
                f"lo mejor obtenido fueron {nb_b} enlaces con grados "
                f"{Counter(len(s) for s in adjB)}. El .gro y el itp no describen "
                f"la misma molécula, o la geometría del .gro está distorsionada "
                f"(¿la molécula quedó partida por la caja y el .gro no está "
                f"centrado?). Revisa la estructura antes de continuar.")

        if use_elements:
            schemes = [("elementos", [itp_el[i] or "X" for i in range(nat)], list(gro_el))]
        else:
            schemes = [("H/pesado", ["H" if e == "H" else "X" for e in itp_el],
                        ["H" if e == "H" else "X" for e in el_for_bonds])]
        # último recurso: solo conectividad, sin usar los nombres del .gro
        schemes.append(("solo conectividad", ["X"] * nat, ["X"] * nat))

        maps, used_scheme = [], None
        for label, initA, initB in schemes:
            uni = {v: k for k, v in enumerate(sorted(set(initA) | set(initB)))}
            labA, labB = wl_labels([adjA, adjB],
                                   [[uni[x] for x in initA], [uni[x] for x in initB]])
            maps = find_isomorphisms(adjA, labA, adjB, labB)
            if maps:
                used_scheme = label
                break
            if label != schemes[-1][0]:
                print(f"[AVISO] No hay isomorfismo usando '{label}'. Los nombres "
                      f"del .gro no son consistentes con su propia geometría. "
                      f"Se reintenta usando solo la conectividad.")
        if not maps:
            die("No se encontró ninguna correspondencia entre el grafo del itp y "
                "el del .gro. Los dos archivos no describen la misma molécula. "
                "Regenera el itp a partir de la misma estructura (mismo .mol2) "
                "que usaste para construir el sistema.")
        if used_scheme == "solo conectividad":
            print("[AVISO] La correspondencia se dedujo SOLO de la conectividad: "
                  "los nombres del .gro no son fiables. Revisa la tabla de abajo "
                  "átomo por átomo antes de usar el resultado.")

        key = "geom_ok" if used_scheme == "solo conectividad" else "ok"
        valid = [(m, validate(list(m))) for m in maps]
        valid = [(m, v) for m, v in valid if v[key]]
        if not valid:
            die("Se encontraron correspondencias topológicas pero ninguna da "
                "longitudes de enlace razonables. Revisa la geometría del .gro.")
        # preferir la que más nombres hace coincidir
        valid.sort(key=lambda mv: -sum(1 for i in range(nat)
                                       if itp_names[i] == gro_names[mv[0][i]]))
        mapping, how = list(valid[0][0]), "grafo"
        agree = sum(1 for i in range(nat) if itp_names[i] == gro_names[mapping[i]])
        print(f"[INFO] {len(valid)} correspondencia(s) válida(s). Elegida la que "
              f"conserva {agree}/{nat} nombres.")

        if len(valid) > 1:
            porgro = defaultdict(set)
            for m, _ in valid:
                for i in range(nat):
                    porgro[m[i]].add((atoms[i]["type"], round(atoms[i]["charge"], 4)))
            ambiguous = [j for j, s in porgro.items() if len(s) > 1]
            if ambiguous:
                print(f"[AVISO] La molécula tiene simetría topológica y "
                      f"{len(ambiguous)} átomo(s) podrían recibir tipo o carga "
                      f"distintos según la solución elegida:")
                for j in sorted(ambiguous)[:10]:
                    opts = sorted(porgro[j])
                    print(f"        {gro_names[j]}: {opts}")
                if not args.accept_ambiguous:
                    die("Asignación ambigua. Revisa el caso y, si las diferencias "
                        "de carga son despreciables, repite con --accept-ambiguous.")

    # ---------------- resultado del emparejamiento ----------------
    v = validate(mapping)
    print(f"\n[OK] Correspondencia establecida por {how}. "
          f"Enlaces del itp medidos sobre el .gro: "
          f"{v['dmin']:.3f}-{v['dmax']:.3f} nm, {len(v['bad'])} anómalos, "
          f"{v['hmis']} H inconsistentes.")

    # new_order[k] = índice 0-based del átomo del itp que va en la posición k
    new_order = [0] * nat
    for i, j in enumerate(mapping):
        new_order[j] = i
    old2new = [0] * nat
    for k, i in enumerate(new_order):
        old2new[i] = k + 1                      # 1-based para el itp nuevo

    out_names = [gro_names[k] if args.names == "gro" else atoms[new_order[k]]["name"]
                 for k in range(nat)]
    n_moved = sum(1 for k, i in enumerate(new_order) if k != i)
    n_renamed = sum(1 for k, i in enumerate(new_order)
                    if atoms[i]["name"] != out_names[k])
    print(f"[INFO] {n_moved}/{nat} átomos cambian de posición; "
          f"{n_renamed} cambian de nombre.")
    if n_moved == 0 and n_renamed == 0:
        print("[OK] El itp ya está en el mismo orden y con los mismos nombres "
              "que el .gro. No hace falta cambiar nada.")
        return
    print("\n  pos | itp original        -> itp nuevo")
    shown = 0
    for k, i in enumerate(new_order):
        if k != i or atoms[i]["name"] != out_names[k]:
            print(f"  {k + 1:4d} | #{i + 1:<4d} {atoms[i]['name']:<6s} "
                  f"-> #{k + 1:<4d} {out_names[k]:<6s} ({atoms[i]['type']}, "
                  f"q={atoms[i]['charge']:+.4f})")
            shown += 1
            if shown >= 40:
                print("  ... (lista truncada)")
                break

    if not args.out:
        print("\n[INFO] Solo diagnóstico. Vuelve a correr con -o salida.itp "
              "para escribir el itp reordenado.")
        return

    # ---------------- reescritura ----------------
    out_lines = []
    it_atom = iter(range(nat))
    pend_sections = defaultdict(list)     # sección -> líneas idx pendientes
    # primera pasada: agrupar las líneas idx por sección para poder ordenarlas
    sect_has_raw = defaultdict(bool)
    for rec in itp.records:
        if rec[2] == mol["name"] and rec[0] == "raw" and rec[1] in SECT_IDX \
                and rec[3].strip().startswith("#"):
            sect_has_raw[rec[1]] = True

    qtot = 0.0
    counters = defaultdict(int)
    for rec in itp.records:
        kind, section, molname, payload = rec
        if molname != mol["name"]:
            out_lines.append(payload if kind != "idx" else payload["raw"])
            continue

        if kind == "atom":
            k = next(it_atom)
            i = new_order[k]
            a = dict(atoms[i])
            if args.names == "gro":
                a["name"] = gro_names[k]
            qtot += a["charge"]
            out_lines.append(fmt_atom(a, k + 1, k + 1, qtot))
            continue

        if kind == "idx":
            new_idx = [old2new[v - 1] for v in payload["idx"]]
            if args.keep_comments:
                com = payload["comment"]
            else:
                names = [(gro_names[v - 1] if args.names == "gro"
                          else atoms[new_order[v - 1]]["name"]) for v in new_idx]
                com = " " + " - ".join(names)
            line = fmt_idx(new_idx, payload["rest"], com)
            if args.no_sort or sect_has_raw[section]:
                out_lines.append(line)
            else:
                pend_sections[section].append((tuple(new_idx), line))
                out_lines.append(("PEND", section, counters[section]))
                counters[section] += 1
            continue

        out_lines.append(payload)

    # volcar las secciones ordenadas en los huecos reservados
    order_map = {}
    for sec, items in pend_sections.items():
        srt = sorted(items, key=lambda t: t[0])
        order_map[sec] = [l for _, l in srt]
    final = []
    for x in out_lines:
        if isinstance(x, tuple) and x and x[0] == "PEND":
            final.append(order_map[x[1]][x[2]])
        else:
            final.append(x)

    # ---------------- verificación posterior ----------------
    with open(args.out, "w") as fh:
        fh.write("\n".join(final) + "\n")

    chk = ItpFile(args.out)
    cmol = chk.get_mol(mol["name"])
    if len(cmol["atoms"]) != nat:
        die(f"El itp escrito tiene {len(cmol['atoms'])} átomos y debería tener {nat}.")
    if [a["nr"] for a in cmol["atoms"]] != list(range(1, nat + 1)):
        die("El itp escrito no quedó numerado 1..N.")
    q0 = round(sum(a["charge"] for a in atoms), 6)
    q1 = round(sum(a["charge"] for a in cmol["atoms"]), 6)
    if abs(q0 - q1) > 1e-6:
        die(f"La carga total cambió: {q0} -> {q1}.")
    n0 = Counter(r[1] for r in itp.records if r[0] == "idx" and r[2] == mol["name"])
    n1 = Counter(r[1] for r in chk.records if r[0] == "idx" and r[2] == mol["name"])
    if n0 != n1:
        die(f"Se perdieron o duplicaron líneas: {n0} -> {n1}.")
    t0 = Counter(a["type"] for a in atoms)
    t1 = Counter(a["type"] for a in cmol["atoms"])
    if t0 != t1:
        die(f"Los tipos de átomo no se conservaron: {t0} -> {t1}.")
    new_bonds = [(i - 1, j - 1) for i, j in cmol["bonds"]]
    d = pair_distances(coords, new_bonds, bvec)
    bad = int(((d < args.lo) | (d > args.hi)).sum())
    if bad:
        die(f"El itp escrito todavía tiene {bad} enlaces con longitud imposible.")
    names_ok = all(cmol["atoms"][k]["name"] == gro_names[k] for k in range(nat)) \
        if args.names == "gro" else True

    print(f"\n[OK] Escrito {args.out}")
    print(f"     {nat} átomos, carga total {q1:+.4f}, secciones {dict(n1)}")
    print(f"     Enlaces medidos sobre el .gro: {d.min():.3f}-{d.max():.3f} nm, "
          f"0 anómalos")
    if names_ok:
        print("     Los nombres coinciden con el .gro: grompp no emitirá warnings "
              "de 'atom name does not match'.")
    print("\n     Recuerda: el .top debe incluir este itp nuevo y, si el "
          "índice del sistema cambió, regenera index.ndx con gmx make_ndx.")


if __name__ == "__main__":
    main()
