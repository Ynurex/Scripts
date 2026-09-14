#!/usr/bin/env python3
"""
itp_charge_viz.py — Visualizacion de cargas parciales (GAFF/AM1-BCC/RESP) de un
ligando parametrizado con antechamber/acpype.

Admite dos modos de entrada:
  A) .itp de GROMACS + .pdb (o .mol2) con la estructura
  B) un unico .mol2 (con cargas), del que se sacan cargas, tipos y estructura

Genera:
  1. <base>_cargas.csv          tabla de cargas por atomo
  2. <base>_cargas.defattr      atributo de ChimeraX (heatmap por carga)
  3. <base>_heatmap.cxc         script de ChimeraX listo para abrir
  4. <base>_cargas.png          grafica de carga por atomo + resumen por tipo

Uso:
    python3 itp_charge_viz.py verteporfina.itp verteporfina.pdb [--attr gaffcharge]
    python3 itp_charge_viz.py verteporfina.mol2
    python3 itp_charge_viz.py verteporfina.itp verteporfina.mol2
"""

import argparse
import csv
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Misma rampa que se usa en el .cxc (azul - blanco - rojo)
CMAP = LinearSegmentedColormap.from_list("bwr_chimerax", ["#0000FF", "#FFFFFF", "#FF0000"])

# Masas atomicas (para el .mol2, que no las trae, y para inferir el elemento)
MASSES = {
    "H": 1.00800, "C": 12.01100, "N": 14.00700, "O": 15.99940, "F": 18.99840,
    "NA": 22.98977, "MG": 24.30500, "P": 30.97376, "S": 32.06000, "CL": 35.45300,
    "K": 39.09830, "CA": 40.07800, "FE": 55.84500, "ZN": 65.38000, "BR": 79.90400,
    "I": 126.90447,
}

# Tipos GAFF (minusculas) de dos letras que SI son un elemento de dos letras.
# Ojo: en GAFF "ca", "na", "os"... son C, N, O — no calcio, sodio ni osmio.
GAFF_TWO_LETTER = {"cl": "CL", "br": "BR"}


# --------------------------------------------------------------------------- #
# Elementos
# --------------------------------------------------------------------------- #
def element_from_mass(mass):
    """Elemento a partir de la masa (lo mas fiable cuando viene en el .itp)."""
    if mass is None or (isinstance(mass, float) and math.isnan(mass)):
        return None
    best, diff = None, 1e9
    for el, m in MASSES.items():
        d = abs(m - mass)
        if d < diff:
            best, diff = el, d
    return best.capitalize() if diff < 0.6 else None


def element_from_type(atype):
    """Elemento a partir del tipo de atomo (GAFF: c3, ha, os, nc; SYBYL: C.3, N.ar)."""
    t = atype.strip()
    if "." in t:                       # SYBYL -> C.3, N.ar, S.o2
        t = t.split(".", 1)[0]
    if not t:
        return "X"
    if len(t) >= 2 and t[0].isupper() and t[1].islower():
        # Escrito como simbolo quimico (Cl, Br, Na, Zn)
        if t[:2].upper() in MASSES:
            return t[:2].capitalize()
    tl = t.lower()
    if tl in GAFF_TWO_LETTER:          # GAFF: cl, br
        return GAFF_TWO_LETTER[tl].capitalize()
    return t[0].upper()


def element_of(atom):
    """Elemento del atomo: elemento explicito > masa > tipo."""
    if atom.get("element"):
        return atom["element"]
    el = element_from_mass(atom.get("mass"))
    if el is None:
        el = element_from_type(atom["type"])
    atom["element"] = el
    return el


def mass_of_element(el):
    return MASSES.get(el.upper(), float("nan"))


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def parse_itp_atoms(path):
    """Devuelve la lista de atomos de la seccion [ atoms ] de un .itp."""
    atoms = []
    section = None
    with open(path) as fh:
        for raw in fh:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"\[\s*(\w+)\s*\]", line)
            if m:
                section = m.group(1).lower()
                continue
            if section != "atoms":
                continue
            f = line.split()
            if len(f) < 7:
                continue
            atoms.append({
                "nr": int(f[0]),
                "type": f[1],
                "resnr": int(f[2]),
                "resname": f[3],
                "name": f[4],
                "cgnr": int(f[5]),
                "charge": float(f[6]),
                "mass": float(f[7]) if len(f) > 7 else float("nan"),
            })
    if not atoms:
        sys.exit(f"ERROR: no se encontro una seccion [ atoms ] con datos en {path}")
    return atoms


def parse_pdb_atoms(path):
    """Devuelve (serial, nombre, resname, resnr, chain) por atomo del PDB."""
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                out.append({
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resnr": int(line[22:26]),
                })
    return out


def parse_mol2_atoms(path):
    """Atomos de la seccion @<TRIPOS>ATOM del primer registro MOLECULE del .mol2.

    Formato:  id  name  x  y  z  type  [subst_id  subst_name  charge]
    Devuelve dicts con las mismas claves que parse_itp_atoms (mass estimada del
    elemento, cgnr = id) mas 'chain' vacio para reutilizar el resto del script.
    """
    atoms = []
    section = None
    n_molecules = 0
    mol_name = None
    mol_header_left = 0
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("@<TRIPOS>"):
                section = s[len("@<TRIPOS>"):].strip().upper()
                if section == "MOLECULE":
                    n_molecules += 1
                    mol_header_left = 1 if n_molecules == 1 else 0
                continue
            if not s or s.startswith("#"):
                continue
            if section == "MOLECULE" and mol_header_left:
                mol_name = s.split()[0]
                mol_header_left = 0
                continue
            if section != "ATOM" or n_molecules > 1:
                continue
            f = s.split()
            if len(f) < 6:
                continue
            if len(f) < 9:
                sys.exit(f"ERROR: el .mol2 {path} no trae cargas en @<TRIPOS>ATOM "
                         "(se necesitan 9 columnas: id name x y z type subst_id subst_name charge)")
            atype = f[5]
            el = element_from_type(atype)
            atoms.append({
                "nr": int(f[0]),
                "type": atype,
                "resnr": int(f[6]),
                "resname": f[7],
                "name": f[1],
                "cgnr": int(f[0]),
                "charge": float(f[8]),
                "mass": mass_of_element(el),
                "element": el,
                "chain": "",
                "serial": int(f[0]),
            })
    if not atoms:
        sys.exit(f"ERROR: no se encontro una seccion @<TRIPOS>ATOM con datos en {path}")
    if n_molecules > 1:
        print(f"AVISO: el .mol2 contiene {n_molecules} moleculas; se usa solo la primera"
              + (f" ({mol_name})" if mol_name else ""))
    return atoms, mol_name


def parse_structure_atoms(path):
    """Nombres/residuos/cadena por atomo a partir de un .pdb o un .mol2."""
    if path.lower().endswith(".mol2"):
        atoms, _ = parse_mol2_atoms(path)
        return [{"serial": a["nr"], "name": a["name"], "resname": a["resname"],
                 "chain": "", "resnr": a["resnr"]} for a in atoms]
    return parse_pdb_atoms(path)


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def write_csv(atoms, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nr", "atom_name", "atom_type", "element", "charge_e", "mass"])
        for a in atoms:
            w.writerow([a["nr"], a["name"], a["type"], element_of(a),
                        f"{a['charge']:.6f}", f"{a['mass']:.5f}"])


def write_defattr(atoms, path, attr_name, struct_atoms, source_note):
    """Archivo de atributos de ChimeraX (una linea por atomo)."""
    chain = ""
    if struct_atoms and struct_atoms[0].get("chain"):
        chain = "/" + struct_atoms[0]["chain"]
    with open(path, "w") as fh:
        fh.write(f"# Cargas parciales del campo de fuerza, extraidas de {source_note}\n")
        fh.write(f"attribute: {attr_name}\n")
        fh.write("match mode: 1-to-1\n")
        fh.write("recipient: atoms\n")
        for a in atoms:
            spec = f"{chain}:{a['resnr']}@{a['name']}"
            fh.write(f"\t{spec}\t{a['charge']:.6f}\n")


CXC_TEMPLATE = """# ===========================================================================
# {title} — heatmap de carga parcial por atomo
# El atributo "{attr}" son las cargas de {source} (unidades de e).
#
#   Uso:   ChimeraX -> File > Open... -> {cxc}
#          o desde la linea de comandos de ChimeraX:  open {cxc}
#   Manten la estructura, el .defattr y este .cxc en la misma carpeta.
# ===========================================================================

close session
set bgColor white

# --- 1. Estructura + atributo de carga --------------------------------------
open {struct}
open {defattr}
# (el .defattr asigna el atributo por residuo/nombre de atomo: :1@C1, :1@C2, ...
#  por eso conviene abrirlo cuando solo esta cargado este modelo)

# --- 2. Representacion -------------------------------------------------------
style ball
size ballScale 0.22 stickRadius 0.13
graphics silhouettes true width 1.5
lighting soft

# --- 3. Heatmap (azul = negativo, blanco = 0, rojo = positivo) ---------------
color byattribute {attr} #1 palette {vmin:.2f},blue:0,white:{vmax:.2f},red

# barra de color + titulo
key blue:{vmin:.2f} white:0 red:{vmax:.2f} pos 0.32,0.06 size 0.36,0.035 fontSize 14 labelColor black showTool false
2dlabels text "carga parcial (e)" xpos 0.42 ypos 0.115 size 15 color black

# --- 4. Etiqueta numerica en los atomos mas polarizados (|q| >= 0.40 e) ------
label {extreme_spec} atoms attribute {attr} height 0.4 color black
# quitar etiquetas:  ~label

view
# guardar imagen:
# save {png_render} width 2400 supersample 3

# ===========================================================================
# EXTRAS — descomenta la linea que quieras usar
# ===========================================================================

# -- Superficie molecular pintada por carga (mapa tipo "electrostatico") -----
# surface #1 resolution 0.4 probeRadius 1.4
# color byattribute {attr} #1 palette {vmin:.2f},blue:0,white:{vmax:.2f},red target s
# transparency #1 45 target s

# -- Solo esqueleto pesado (oculta hidrogenos) -------------------------------
# hide @@element=H target ab

# -- Escala robusta por percentiles 5-95 (mas contraste en el rango comun) ---
# color byattribute {attr} #1 palette {p5:.2f},blue:0,white:{p95:.2f},red

# -- Etiquetar TODOS los atomos con su carga ---------------------------------
# label #1 atoms attribute {attr} height 0.3

# -- Ver la carga de un atomo con click derecho ------------------------------
# mousemode right label

# -- Listar valores / comprobar la carga neta --------------------------------
# info atoms #1 attribute {attr}
"""



def write_cxc(atoms, path, attr_name, struct_name, defattr_name, title, png_render,
              source_note):
    q = np.array([a["charge"] for a in atoms])
    lim = float(np.max(np.abs(q)))
    lim = np.ceil(lim * 20) / 20.0  # redondeo a 0.05
    p5, p95 = np.percentile(q, 5), np.percentile(q, 95)
    pl = max(abs(p5), abs(p95))

    # atomos por encima de |0.4 e| -> se etiquetan
    extremes = [a["name"] for a in atoms if abs(a["charge"]) >= 0.40]
    spec = "#1@" + ",".join(extremes) if extremes else "#1"

    with open(path, "w") as fh:
        fh.write(CXC_TEMPLATE.format(
            title=title, attr=attr_name, struct=struct_name, defattr=defattr_name,
            cxc=os.path.basename(path), vmin=-lim, vmax=lim,
            p5=-pl, p95=pl, extreme_spec=spec, png_render=png_render,
            source=source_note,
        ))
    return lim


def plot_charges(atoms, path, lim, title, type_label="GAFF"):
    q = np.array([a["charge"] for a in atoms])
    names = [a["name"] for a in atoms]
    types = [a["type"] for a in atoms]
    elems = [element_of(a) for a in atoms]
    norm = Normalize(vmin=-lim, vmax=lim)
    colors = CMAP(norm(q))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1], width_ratios=[2.6, 1],
                          hspace=0.42, wspace=0.18)

    # ---- (a) carga por atomo ------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(q))
    ax.bar(x, q, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.8)
    for lv in (0.4, -0.4):
        ax.axhline(lv, color="grey", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_xlim(-1, len(q))
    ax.set_ylabel("carga parcial (e)", fontsize=11)
    ax.set_title(f"{title} — carga parcial por átomo   "
                 f"(Σq = {q.sum():+.4f} e, n = {len(q)})",
                 fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # anotar los tres mas positivos y los tres mas negativos
    for i in list(np.argsort(q)[:3]) + list(np.argsort(q)[-3:]):
        off = 0.045 if q[i] > 0 else -0.045
        ax.annotate(f"{names[i]}\n{q[i]:+.3f}", (i, q[i] + off),
                    ha="center", va="bottom" if q[i] > 0 else "top",
                    fontsize=7.5, color="#222222")
    ax.set_ylim(min(q) - 0.20, max(q) + 0.20)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, pad=0.008, aspect=28)
    cb.set_label("q (e)", fontsize=10)

    # ---- (b) distribucion por tipo de atomo ---------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    order = sorted(set(types), key=lambda t: np.mean([a["charge"] for a in atoms if a["type"] == t]))
    for j, t in enumerate(order):
        vals = np.array([a["charge"] for a in atoms if a["type"] == t])
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.28
        ax2.scatter(np.full(len(vals), j) + jitter, vals, s=26,
                    c=CMAP(norm(vals)), edgecolor="#333333", linewidth=0.4, zorder=3)
        ax2.hlines(vals.mean(), j - 0.32, j + 0.32, color="#111111", lw=1.6, zorder=4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, fontsize=9, rotation=90 if len(order) > 14 else 0)
    ax2.set_xlabel(f"tipo de átomo {type_label}", fontsize=10)
    ax2.set_ylabel("carga parcial (e)", fontsize=10)
    ax2.set_title("Distribución por tipo de átomo (barra = media)", fontsize=11)
    ax2.grid(axis="y", alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)

    # ---- (c) carga acumulada por elemento ----------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    el_order = ["C", "N", "O", "H", "S", "P", "F", "Cl", "Br", "I"]
    el_order = [e for e in el_order if e in elems] + \
               [e for e in sorted(set(elems)) if e not in el_order]
    sums = [sum(a["charge"] for a in atoms if element_of(a) == e) for e in el_order]
    counts = [sum(1 for a in atoms if element_of(a) == e) for e in el_order]
    bars = ax3.bar(el_order, sums, color=["#d96b6b" if v > 0 else "#6b86d9" for v in sums],
                   edgecolor="#333333", linewidth=0.6)
    for b, s, n in zip(bars, sums, counts):
        ax3.annotate(f"{s:+.2f}\n(n={n})", (b.get_x() + b.get_width() / 2, s),
                     ha="center", va="bottom" if s > 0 else "top", fontsize=8.5)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("Σ carga (e)", fontsize=10)
    ax3.set_title("Carga total por elemento", fontsize=11)
    ax3.grid(axis="y", alpha=0.25, lw=0.6)
    ax3.set_axisbelow(True)
    m = max(abs(min(sums)), abs(max(sums))) or 1.0
    ax3.set_ylim(-m * 1.45, m * 1.45)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def classify_inputs(paths):
    """Separa los archivos de entrada en (itp, estructura) segun su extension."""
    itp = struct = None
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".itp", ".top"):
            if itp:
                sys.exit("ERROR: se han dado dos topologias (.itp); solo se admite una.")
            itp = p
        elif ext in (".pdb", ".ent", ".mol2"):
            if struct:
                sys.exit("ERROR: se han dado dos estructuras (.pdb/.mol2); solo se admite una.")
            struct = p
        else:
            sys.exit(f"ERROR: extension no reconocida en {p} (se espera .itp, .pdb o .mol2)")
    if itp is None and struct is None:
        sys.exit("ERROR: no se ha dado ningun archivo de entrada.")
    if itp is not None and struct is None:
        sys.exit("ERROR: con un .itp hay que dar tambien el .pdb (o .mol2) de la estructura.")
    if itp is None and not struct.lower().endswith(".mol2"):
        sys.exit("ERROR: con un solo archivo debe ser un .mol2 con cargas "
                 "(un .pdb no contiene cargas).")
    return itp, struct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", metavar="ARCHIVO",
                    help="un .mol2 con cargas, o un .itp junto con su .pdb/.mol2")
    ap.add_argument("--attr", default="gaffcharge",
                    help="nombre del atributo en ChimeraX (default: gaffcharge)")
    ap.add_argument("--base", default=None, help="prefijo de los archivos de salida")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    itp_path, struct_path = classify_inputs(args.inputs)
    for p in (itp_path, struct_path):
        if p and not os.path.isfile(p):
            sys.exit(f"ERROR: no existe el archivo {p}")

    mol_name = None
    if itp_path:
        atoms = parse_itp_atoms(itp_path)
        struct_atoms = parse_structure_atoms(struct_path)
        source_note = os.path.basename(itp_path)
        type_label = "GAFF"

        # --- verificacion de correspondencia topologia <-> estructura --------
        if len(atoms) != len(struct_atoms):
            print(f"AVISO: el .itp tiene {len(atoms)} átomos y "
                  f"{os.path.basename(struct_path)} {len(struct_atoms)}.")
        mismatch = [(a["nr"], a["name"], p["name"])
                    for a, p in zip(atoms, struct_atoms) if a["name"] != p["name"]]
        if mismatch:
            print(f"AVISO: {len(mismatch)} nombres no coinciden entre el .itp y la "
                  f"estructura, p.ej. {mismatch[:5]}")
        else:
            print(f"OK: {len(atoms)} átomos, nombres del .itp y de la estructura "
                  "coinciden 1 a 1.")

        # el residuo de la estructura manda para los specs de ChimeraX
        for a, p in zip(atoms, struct_atoms):
            a["resnr"] = p["resnr"]
    else:
        atoms, mol_name = parse_mol2_atoms(struct_path)
        struct_atoms = [{"serial": a["nr"], "name": a["name"], "resname": a["resname"],
                         "chain": "", "resnr": a["resnr"]} for a in atoms]
        source_note = os.path.basename(struct_path)
        # antechamber escribe tipos GAFF en el mol2; SYBYL lleva punto (C.3, N.ar)
        type_label = "SYBYL" if any("." in a["type"] for a in atoms) else "GAFF"
        dup = len(atoms) - len({(a["resnr"], a["name"]) for a in atoms})
        if dup:
            print(f"AVISO: {dup} nombres de átomo repetidos en el .mol2; el .defattr "
                  "puede asignar la carga al átomo equivocado en ChimeraX.")
        print(f"OK: {len(atoms)} átomos leídos del .mol2 "
              f"(tipos {type_label}, cargas de la columna 9).")

    base = args.base or os.path.splitext(os.path.basename(itp_path or struct_path))[0]
    os.makedirs(args.outdir, exist_ok=True)
    o = lambda s: os.path.join(args.outdir, s)
    csv_p = o(f"{base}_cargas.csv")
    att_p = o(f"{base}_cargas.defattr")
    cxc_p = o(f"{base}_heatmap.cxc")
    png_p = o(f"{base}_cargas.png")

    write_csv(atoms, csv_p)
    write_defattr(atoms, att_p, args.attr, struct_atoms, source_note)
    lim = write_cxc(atoms, cxc_p, args.attr, os.path.basename(struct_path),
                    os.path.basename(att_p), base, f"{base}_render.png", source_note)
    plot_charges(atoms, png_p, lim, base, type_label)

    q = np.array([a["charge"] for a in atoms])
    print(f"Carga neta        : {q.sum():+.6f} e")
    print(f"Rango             : {q.min():+.4f} .. {q.max():+.4f} e (escala ±{lim:.2f})")
    print("Escritos:", ", ".join(map(os.path.basename, [csv_p, att_p, cxc_p, png_p])))


if __name__ == "__main__":
    main()
