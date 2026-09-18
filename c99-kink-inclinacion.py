#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helix_kink_tilt.py
==================

Calcula, a lo largo de una trayectoria de MD, dos observables geometricos de una
helice transmembranal (p. ej. el dominio TM de C99 en una bicapa de POPC):

  1) KINK (angulo de codo/bend): angulo entre el eje de un segmento N-terminal y
     el eje de un segmento C-terminal de la misma helice, ambos orientados N->C.
     Convencion: 0 grados = helice recta. (El "angulo interno" suplementario
     tambien se reporta: 180 - kink.)

     Con --kink-resid se puede pedir el angulo en residuos bisagra concretos
     (p. ej. --kink-resid 17,21,25). Para cada residuo r se ajusta un eje a la
     ventana de CA que lo precede y otro a la que lo sigue (excluyendo r y sus
     vecinos inmediatos) y se reporta el angulo entre ambos, frame a frame.
     En paralelo se extrae el bend local de HELANAL centrado en ese mismo
     residuo, como medida independiente del mismo codo.

  2) TILT (inclinacion): angulo entre el eje de la helice (o de cada segmento) y
     la normal de la membrana. Se reporta el angulo crudo en [0,180] respecto a
     la normal orientada hacia +z, y el angulo "plegado" en [0,90].

El eje de cada segmento se obtiene por ajuste de minimos cuadrados (SVD) sobre
las coordenadas de los C-alfa, o sobre los "origenes locales" de HELANAL
(recomendado: elimina el bamboleo helicoidal de +-2.3 A de los CA).
Opcionalmente se corre HELANAL (MDAnalysis.analysis.helix_analysis) como control
independiente: da el tilt global, el perfil de bends locales (util para
localizar la bisagra) y los giros locales (control de helicidad).

Disenado para ser generico: sirve para cualquier proteina/helice, cualquier
lipido y cualquier normal, siempre que se den las selecciones adecuadas.

Verificaciones internas automaticas
-----------------------------------
  * Coherencia topologia/trayectoria (numero de atomos, dimensiones de caja).
  * Selecciones no vacias; exactamente un CA por residuo (detecta segid/altloc
    duplicados, que es el error tipico al mezclar cadenas de CHARMM-GUI).
  * Presencia y contigüidad de los resid pedidos (reporta residuos faltantes).
  * Segmentos no solapados, con numero minimo de residuos para ajustar un eje.
  * Reconstruccion de la cadena rota por PBC (minimum image a lo largo del
     backbone) y aviso de cuantos frames necesitaron correccion.
  * Caja ortorrombica (requisito de la reconstruccion PBC y del split de valvas).
  * Bicapa partida por PBC en z: se recentra con media circular antes de separar
    valvas; se avisa si las valvas estan desbalanceadas o el grosor P-P es raro.
  * Linealidad de cada segmento (s2/s1 del SVD): avisa si un segmento no es
    suficientemente recto como para que "un eje" lo describa.
  * Rise por residuo (~1.5 A en helice alfa) como control de helicidad.

Salidas
-------
  <prefix>_timeseries.csv   serie temporal por frame
  <prefix>_summary.json     metadatos, argumentos, checks y estadisticas
  <prefix>.png              graficas (si matplotlib esta disponible)
  <prefix>_helanal.csv      (opcional) perfil promedio de bends/twists locales
  <prefix>_kinks.png        (opcional) angulos en los residuos de --kink-resid

Ejemplo
-------
  python helix_kink_tilt.py -s md.tpr -f md_center.xtc \
      --resid-range 700-723 --nter 700-707 --cter 712-723 \
      --normal-mode leaflets --axis-method origins --helanal -o c99_tm

  # angulo de kink en residuos concretos (implica --helanal):
  python helix_kink_tilt.py -s md.tpr -f md_center.xtc \
      --resid-range 1-40 --kink-resid 17,21,25 --kink-window 7 --kink-gap 1 \
      -o c99_kinks

Autor: script generado para trabajo de simulacion de membranas (MDAnalysis >= 2.0).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import helix_analysis as hel
except ImportError as exc:  # pragma: no cover
    sys.exit("ERROR: se requiere MDAnalysis >= 2.0 (pip install MDAnalysis). %s" % exc)


# --------------------------------------------------------------------------- #
# Utilidades de mensajes
# --------------------------------------------------------------------------- #
CHECKS = []          # lista de (nivel, mensaje) para el JSON de salida


def _log(level, msg):
    CHECKS.append((level, msg))
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print("[%-5s] %s" % (level, msg), file=stream)


def info(msg):
    _log("INFO", msg)


def warn(msg):
    _log("WARN", msg)


def die(msg):
    _log("ERROR", msg)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #
def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Vector de norma ~0; no se puede normalizar.")
    return v / n


def angle_deg(v1, v2, fold=False):
    """Angulo entre dos vectores en grados. fold=True lo pliega a [0,90]."""
    c = np.clip(np.dot(unit(v1), unit(v2)), -1.0, 1.0)
    a = np.degrees(np.arccos(c))
    return 180.0 - a if (fold and a > 90.0) else a


def best_fit_axis(coords):
    """
    Eje principal por SVD sobre coordenadas (N,3), orientado del primer al
    ultimo punto (es decir, N->C si las coordenadas van en orden de secuencia).

    Devuelve (eje_unitario, linealidad) donde linealidad = s2/s1 (0 = recta
    perfecta; valores altos indican que el segmento no es lineal).
    """
    coords = np.asarray(coords, dtype=float)
    centered = coords - coords.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if np.dot(axis, coords[-1] - coords[0]) < 0:
        axis = -axis
    linearity = float(s[1] / s[0]) if s[0] > 1e-9 else np.nan
    return unit(axis), linearity


def helix_axis(coords, method="origins", ref_axis=(0, 0, 1)):
    """
    Eje de una helice.
      method='ca'      -> SVD sobre los CA.
      method='origins' -> SVD sobre los origenes locales de HELANAL (quita el
                          bamboleo helicoidal de +-2.3 A). Requiere >= 6 CA; si
                          hay menos, cae a 'ca'.

    Nota: sobre helices ideales, 'origins' recupera el eje exacto incluso con
    segmentos de 8 residuos, mientras que 'ca' puede errar varios grados.
    """
    coords = np.asarray(coords, dtype=float)
    if method == "origins" and len(coords) >= 6:
        res = hel.helix_analysis(coords, ref_axis=np.asarray(ref_axis, dtype=float))
        origins = np.asarray(res["local_origins"], dtype=float)
        if len(origins) >= 3:
            return best_fit_axis(origins)
    return best_fit_axis(coords)


def make_whole_chain(pos, box_lengths):
    """
    Reconstruye una cadena partida por PBC aplicando minimum image entre atomos
    consecutivos (valido para cajas ortorrombicas). Devuelve (pos_corregida,
    n_saltos_corregidos).
    """
    out = np.array(pos, dtype=float, copy=True)
    L = np.asarray(box_lengths, dtype=float)
    jumps = 0
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        shift = np.round(d / L)
        if np.any(shift != 0):
            jumps += 1
        out[i] = out[i - 1] + (d - shift * L)
    return out, jumps


def circular_recenter_z(z, Lz):
    """
    Recentra coordenadas z tratandolas como angulos (media circular), de modo que
    una bicapa partida por la frontera periodica quede entera y centrada en Lz/2.
    Devuelve z recentrada (el desplazamiento es uniforme y no afecta diferencias).
    """
    theta = 2.0 * np.pi * z / Lz
    mean_theta = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    center = (mean_theta % (2.0 * np.pi)) * Lz / (2.0 * np.pi)
    return (z + (Lz / 2.0 - center)) % Lz


def plane_normal(coords):
    """Normal de un plano ajustado por SVD (vector singular menor)."""
    centered = np.asarray(coords, float) - np.mean(coords, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return unit(vt[-1])


def membrane_normal(p_positions, box, mode="leaflets"):
    """
    Normal de la membrana para un frame.
      'z'        -> [0,0,1] (rapido; valido si el sistema no rota, p. ej. con
                    restricciones o tras alinear la bicapa).
      'leaflets' -> vector entre los centroides de las dos valvas (P superior -
                    P inferior). Robusto y barato.
      'plane'    -> promedio de las normales de los planos ajustados a cada valva.
    Devuelve (normal_orientada_hacia_+z, dict_de_diagnostico).
    """
    diag = {}
    if mode == "z":
        return np.array([0.0, 0.0, 1.0]), diag

    pos = np.array(p_positions, dtype=float, copy=True)
    Lz = float(box[2])
    pos[:, 2] = circular_recenter_z(pos[:, 2], Lz)
    zmid = np.median(pos[:, 2])
    up = pos[pos[:, 2] > zmid]
    lo = pos[pos[:, 2] <= zmid]
    diag["n_upper"] = int(len(up))
    diag["n_lower"] = int(len(lo))
    if len(up) < 3 or len(lo) < 3:
        raise ValueError("Valvas con menos de 3 atomos de referencia; revisa --lipid-sel.")
    diag["thickness_PP"] = float(up[:, 2].mean() - lo[:, 2].mean())

    if mode == "leaflets":
        n = unit(up.mean(axis=0) - lo.mean(axis=0))
    elif mode == "plane":
        n_up = plane_normal(up)
        n_lo = plane_normal(lo)
        if n_up[2] < 0:
            n_up = -n_up
        if n_lo[2] < 0:
            n_lo = -n_lo
        n = unit(n_up + n_lo)
    else:
        raise ValueError("normal-mode desconocido: %s" % mode)

    if n[2] < 0:
        n = -n
    return n, diag


# --------------------------------------------------------------------------- #
# Parsing / seleccion / checks
# --------------------------------------------------------------------------- #
def parse_range(text):
    """'700-723', '700:723' o '700 723' -> (700, 723)."""
    t = text.replace(":", "-").replace(",", "-").replace(" ", "-")
    parts = [p for p in t.split("-") if p != ""]
    if len(parts) == 1:
        a = b = int(parts[0])
    elif len(parts) == 2:
        a, b = int(parts[0]), int(parts[1])
    else:
        raise argparse.ArgumentTypeError("Rango invalido: %r (usa 700-723)" % text)
    if b < a:
        a, b = b, a
    return (a, b)


def parse_resid_list(text):
    """'17,21,25', '17 21 25' o '17:21:25' -> [17, 21, 25] (ordenado, sin repetir)."""
    t = text.replace(":", ",").replace(";", ",").replace(" ", ",")
    parts = [p for p in t.split(",") if p != ""]
    try:
        vals = sorted({int(p) for p in parts})
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Lista de resid invalida: %r (usa 17,21,25)" % text)
    if not vals:
        raise argparse.ArgumentTypeError("Lista de resid vacia: %r" % text)
    return vals


def select_ca(universe, resid_range, extra_sel, ca_name):
    """Selecciona los CA del rango pedido y valida el resultado."""
    lo, hi = resid_range
    sel = "name %s and resid %d:%d" % (ca_name, lo, hi)
    if extra_sel:
        sel = "(%s) and (%s)" % (sel, extra_sel)
    ag = universe.select_atoms(sel)
    if ag.n_atoms == 0:
        die("Seleccion vacia: '%s'. Revisa numeracion de resid, nombre de atomo "
            "(--ca-name) y --select-extra (p.ej. 'segid PROA')." % sel)

    resids = ag.resids
    uniq, counts = np.unique(resids, return_counts=True)
    if np.any(counts > 1):
        dup = uniq[counts > 1]
        die("Hay mas de un '%s' por residuo en %s (resids duplicados: %s). "
            "Probablemente seleccionaste varias cadenas/segmentos o hay altloc. "
            "Usa --select-extra \"segid PROA\" (o chainID/altloc) para desambiguar."
            % (ca_name, sel, dup[:10]))

    if not np.all(np.diff(resids) > 0):
        order = np.argsort(resids)
        ag = ag[order]
        warn("Los CA no venian ordenados por resid; se reordenaron. Verifica que "
             "el orden en el archivo corresponda a la secuencia real.")
        resids = ag.resids

    esperados = set(range(lo, hi + 1))
    faltantes = sorted(esperados - set(resids.tolist()))
    if faltantes:
        warn("Residuos ausentes en el rango %d-%d: %s. El ajuste del eje usa solo "
             "los presentes (huecos grandes distorsionan el eje)." % (lo, hi, faltantes))
    info("Seleccion '%s': %d CA (resid %d-%d)." % (sel, ag.n_atoms, resids[0], resids[-1]))
    return ag


def check_box(universe):
    dims = universe.dimensions
    if dims is None or np.any(dims[:3] <= 0):
        die("La trayectoria no trae dimensiones de caja. Necesarias para PBC y "
            "para el calculo de la normal. Usa un XTC/TRR con caja o pasa --normal-mode z "
            "y --no-make-whole bajo tu propio riesgo.")
    angles = dims[3:]
    if not np.allclose(angles, 90.0, atol=1.0):
        warn("Caja no ortorrombica (angulos %s). La reconstruccion PBC de la cadena y "
             "el split de valvas asumen ortorrombica. Para caja dodecaedrica/truncada, "
             "convierte antes: gmx trjconv -pbc mol -ur rect (o -box)." % np.round(angles, 2))
    return dims


def split_segments(resids, args):
    """Determina los rangos (nter, cter) a partir de argumentos o por defecto."""
    lo, hi = int(resids[0]), int(resids[-1])
    if args.nter and args.cter:
        n_rng, c_rng = args.nter, args.cter
    else:
        n_res = len(resids)
        gap = max(0, args.gap)
        half = (n_res - gap) // 2
        if half < args.min_seg:
            die("El rango %d-%d es demasiado corto para partirlo en dos segmentos de "
                "al menos %d residuos (con --gap %d). Da --nter y --cter explicitos."
                % (lo, hi, args.min_seg, gap))
        n_rng = (int(resids[0]), int(resids[half - 1]))
        c_rng = (int(resids[half + gap]), int(resids[-1]))
        info("Segmentos por defecto (mitades con gap=%d): N-ter %d-%d, C-ter %d-%d."
             % (gap, n_rng[0], n_rng[1], c_rng[0], c_rng[1]))
    if n_rng[1] >= c_rng[0]:
        die("Los segmentos se solapan: N-ter %s y C-ter %s. El kink solo tiene sentido "
            "con dos tramos disjuntos separados por la bisagra." % (n_rng, c_rng))
    return n_rng, c_rng


def resolve_kink_sites(resids, args):
    """
    Traduce los residuos de --kink-resid en pares de ventanas flanqueantes.

    Para un residuo bisagra r se ajusta un eje a los <kink-window> CA que lo
    preceden y otro a los <kink-window> que lo siguen, excluyendo r y los
    +-<kink-gap> residuos contiguos: el codo deforma la vuelta de helice a cada
    lado, y meterla en el ajuste contamina los dos ejes a la vez. El kink local
    es el angulo entre ambos ejes (0 grados = tramo recto).

    Devuelve una lista de dicts con los indices (dentro de la seleccion de CA)
    de cada ventana.
    """
    if not args.kink_resid:
        return []
    resids = np.asarray(resids)
    win = max(1, args.kink_window)
    gap = max(0, args.kink_gap)
    sites = []
    for r in args.kink_resid:
        hit = np.where(resids == r)[0]
        if len(hit) == 0:
            die("El residuo bisagra %d no esta en la seleccion (resid %d-%d). Revisa "
                "--kink-resid y que --resid-range cubra la bisagra con margen para las "
                "ventanas." % (r, resids[0], resids[-1]))
        k = int(hit[0])
        idx_n = np.arange(max(0, k - gap - win), max(0, k - gap))
        idx_c = np.arange(min(len(resids), k + gap + 1),
                          min(len(resids), k + gap + 1 + win))
        if len(idx_n) < 4 or len(idx_c) < 4:
            die("El residuo bisagra %d deja solo %d CA antes y %d despues (hacen falta "
                ">=4 por lado, ~1 vuelta de helice, con --kink-window %d y --kink-gap %d). "
                "Amplia --resid-range o reduce la ventana."
                % (r, len(idx_n), len(idx_c), win, gap))
        if len(idx_n) < win or len(idx_c) < win:
            warn("Residuo bisagra %d: ventanas recortadas por el borde del rango "
                 "(%d CA antes, %d despues; se pidieron %d por lado)."
                 % (r, len(idx_n), len(idx_c), win))
        if min(len(idx_n), len(idx_c)) < 6 and args.axis_method == "origins":
            warn("Residuo bisagra %d: alguna ventana tiene <6 CA, asi que su eje se "
                 "ajusta sobre los CA y no sobre origenes locales; el bamboleo "
                 "helicoidal puede sesgar el angulo varios grados." % r)
        site = {"resid": int(r), "idx_n": idx_n, "idx_c": idx_c,
                "nter": (int(resids[idx_n[0]]), int(resids[idx_n[-1]])),
                "cter": (int(resids[idx_c[0]]), int(resids[idx_c[-1]]))}
        sites.append(site)
        info("Kink en resid %d: ventana N %d-%d (%d CA) vs ventana C %d-%d (%d CA)."
             % (r, site["nter"][0], site["nter"][1], len(idx_n),
                site["cter"][0], site["cter"][1], len(idx_c)))

    # Bisagras demasiado juntas: las ventanas de una engloban a otra y el angulo
    # medido deja de ser el de un solo codo.
    pedidos = set(args.kink_resid)
    for site in sites:
        vecinos = sorted(pedidos & set(resids[np.r_[site["idx_n"], site["idx_c"]]].tolist()))
        if vecinos:
            warn("Las ventanas del resid %d contienen tambien la(s) bisagra(s) %s: el "
                 "angulo 'kink_r%d_deg' suma varios codos, no uno solo. Con bisagras "
                 "separadas menos de %d residuos usa --kink-window mas corta (>=4 CA por "
                 "lado) o quedate con el bend local de HELANAL, cuya ventana es de 4 CA."
                 % (site["resid"], vecinos, site["resid"], win + gap + 1))
    return sites


# --------------------------------------------------------------------------- #
# Analisis principal
# --------------------------------------------------------------------------- #
def run_analysis(args):
    info("MDAnalysis %s | numpy %s | python %s"
         % (mda.__version__, np.__version__, platform.python_version()))

    if args.kink_resid and not args.helanal:
        args.helanal = True
        info("--kink-resid activa --helanal: el bend local de HELANAL en cada residuo "
             "pedido sirve de control independiente del angulo entre ventanas.")

    # ---- Universe -------------------------------------------------------- #
    try:
        u = mda.Universe(args.topology, *args.trajectory) if args.trajectory \
            else mda.Universe(args.topology)
    except ValueError as exc:
        die("No se pudo construir el Universe (tipico: el numero de atomos de la "
            "topologia y de la trayectoria no coincide). Detalle: %s" % exc)
    info("Sistema: %d atomos, %d frames." % (u.atoms.n_atoms, u.trajectory.n_frames))

    dims = check_box(u)
    info("Caja del primer frame: %s A" % np.round(dims[:3], 2))

    # ---- Selecciones ----------------------------------------------------- #
    full = select_ca(u, args.resid_range, args.select_extra, args.ca_name)
    resids = full.resids
    n_rng, c_rng = split_segments(resids, args)

    idx_n = np.where((resids >= n_rng[0]) & (resids <= n_rng[1]))[0]
    idx_c = np.where((resids >= c_rng[0]) & (resids <= c_rng[1]))[0]
    for name, idx in (("N-ter", idx_n), ("C-ter", idx_c)):
        if len(idx) < args.min_seg:
            die("El segmento %s tiene solo %d CA (minimo %d). Un eje ajustado a menos de "
                "~1 vuelta de helice (>=4-5 residuos) no es confiable." % (name, len(idx), args.min_seg))
        if len(idx) < 6 and args.axis_method == "origins":
            warn("El segmento %s tiene %d CA (<6): se usara SVD sobre CA en lugar de "
                 "origenes locales. Con segmentos tan cortos el bamboleo helicoidal "
                 "puede sesgar el eje varios grados." % (name, len(idx)))

    kink_sites = resolve_kink_sites(resids, args)

    lipids = None
    if args.normal_mode != "z":
        lipids = u.select_atoms(args.lipid_sel)
        if lipids.n_atoms < 6:
            die("Seleccion de lipidos '%s' con %d atomos. Para POPC de CHARMM-GUI el "
                "default 'name P' suele funcionar; en otros campos de fuerza prueba "
                "'name P8' o 'name P31'." % (args.lipid_sel, lipids.n_atoms))
        info("Referencia de membrana: '%s' -> %d atomos." % (args.lipid_sel, lipids.n_atoms))

    # ---- Loop sobre la trayectoria --------------------------------------- #
    rows = []
    n_jumps_frames = 0
    lin_warned = False
    box_lengths_0 = dims[:3].copy()

    for ts in u.trajectory[args.first:args.last:args.stride]:
        L = ts.dimensions[:3]
        pos = full.positions.copy()

        if not args.no_make_whole:
            pos, jumps = make_whole_chain(pos, L)
            if jumps:
                n_jumps_frames += 1

        # normal de la membrana
        if args.normal_mode == "z":
            normal, diag = np.array([0.0, 0.0, 1.0]), {}
        else:
            normal, diag = membrane_normal(lipids.positions, ts.dimensions, args.normal_mode)

        # ejes
        ax_full, lin_full = helix_axis(pos, args.axis_method, normal)
        ax_n, lin_n = helix_axis(pos[idx_n], args.axis_method, normal)
        ax_c, lin_c = helix_axis(pos[idx_c], args.axis_method, normal)

        if not lin_warned and max(lin_n, lin_c) > args.linearity_cutoff:
            warn("Frame %d: linealidad s2/s1 = %.3f (N) / %.3f (C), por encima de %.2f. "
                 "Uno de los segmentos no es recto: considera acortarlo o mover la bisagra "
                 "(mira el perfil de bends de HELANAL con --helanal)."
                 % (ts.frame, lin_n, lin_c, args.linearity_cutoff))
            lin_warned = True

        kink = angle_deg(ax_n, ax_c)                 # 0 = recta
        rise = float(np.dot(pos[-1] - pos[0], ax_full) / max(len(pos) - 1, 1))

        # kink en cada residuo bisagra pedido con --kink-resid
        kink_local = []
        for site in kink_sites:
            a_n, _ = helix_axis(pos[site["idx_n"]], args.axis_method, normal)
            a_c, _ = helix_axis(pos[site["idx_c"]], args.axis_method, normal)
            kink_local.append(angle_deg(a_n, a_c))

        rows.append([
            ts.frame, float(ts.time),
            kink, 180.0 - kink,
            angle_deg(ax_full, normal), angle_deg(ax_full, normal, fold=True),
            angle_deg(ax_n, normal), angle_deg(ax_n, normal, fold=True),
            angle_deg(ax_c, normal), angle_deg(ax_c, normal, fold=True),
            normal[0], normal[1], normal[2],
            lin_full, lin_n, lin_c, rise,
            diag.get("thickness_PP", np.nan),
        ] + kink_local)

    if not rows:
        die("No se analizo ningun frame. Revisa --first/--last/--stride.")

    data = np.array(rows, dtype=float)
    columns = ["frame", "time_ps", "kink_deg", "kink_internal_deg",
               "tilt_full_deg", "tilt_full_folded_deg",
               "tilt_nter_deg", "tilt_nter_folded_deg",
               "tilt_cter_deg", "tilt_cter_folded_deg",
               "normal_x", "normal_y", "normal_z",
               "linearity_full", "linearity_nter", "linearity_cter",
               "rise_per_res_A", "thickness_PP_A"]
    columns += ["kink_r%d_deg" % s["resid"] for s in kink_sites]

    # ---- Diagnosticos globales ------------------------------------------- #
    if n_jumps_frames:
        warn("%d de %d frames tenian la cadena partida por PBC y se reconstruyeron "
             "(minimum image). Es normal si no corriste 'gmx trjconv -pbc mol'; si el "
             "numero es alto conviene arreglar la trayectoria antes."
             % (n_jumps_frames, len(rows)))
    rise_mean = np.nanmean(data[:, columns.index("rise_per_res_A")])
    if not (1.2 <= rise_mean <= 1.8):
        warn("Rise medio por residuo = %.2f A (helice alfa ideal ~1.5 A). El tramo "
             "probablemente no es helicoidal continuo o hay residuos faltantes." % rise_mean)
    else:
        info("Rise medio por residuo = %.2f A (consistente con helice alfa)." % rise_mean)

    if args.normal_mode != "z":
        th = data[:, columns.index("thickness_PP_A")]
        if np.nanmean(th) < 20 or np.nanmean(th) > 60:
            warn("Distancia media P-P entre valvas = %.1f A (POPC ~ 39 A). Revisa "
                 "--lipid-sel o si la bicapa esta bien centrada." % np.nanmean(th))
        else:
            info("Distancia media P-P entre valvas = %.1f A." % np.nanmean(th))
        nz = data[:, columns.index("normal_z")]
        if np.nanmin(nz) < 0.9:
            warn("La normal se desvia hasta %.1f grados de +z. Normal si la bicapa ondula "
                 "o el sistema rota; si es mucho, revisa el centrado."
                 % np.degrees(np.arccos(np.nanmin(nz))))

    # ---- HELANAL (control independiente) --------------------------------- #
    helanal_out = None
    if args.helanal:
        n_ca = full.n_atoms
        if n_ca < 9:
            warn("HELANAL requiere >= 9 CA; se omite (hay %d)." % n_ca)
        else:
            mean_normal = unit(data[:, [columns.index("normal_x"),
                                        columns.index("normal_y"),
                                        columns.index("normal_z")]].mean(axis=0))
            sel = "name %s and resid %d:%d" % (args.ca_name, args.resid_range[0], args.resid_range[1])
            if args.select_extra:
                sel = "(%s) and (%s)" % (sel, args.select_extra)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = hel.HELANAL(u, select=sel, ref_axis=mean_normal,
                                flatten_single_helix=True).run(
                    start=args.first, stop=args.last, step=args.stride)
            bends = np.asarray(h.results.local_bends)      # (frames, n_windows)
            twists = np.asarray(h.results.local_twists)
            tilts = np.asarray(h.results.global_tilts)
            off_b = max((n_ca - bends.shape[1]) // 2, 0)
            off_t = max((n_ca - twists.shape[1]) // 2, 0)
            folded = np.minimum(tilts, 180.0 - tilts)
            helanal_out = {
                "global_tilt_folded_mean": float(np.mean(folded)),
                "global_tilt_mean": float(np.mean(tilts)),
                "global_tilt_std": float(np.std(tilts)),
                "ref_axis": mean_normal.tolist(),
                "bend_resid": resids[off_b:off_b + bends.shape[1]].tolist(),
                "bend_mean": np.mean(bends, axis=0).tolist(),
                "bend_std": np.std(bends, axis=0).tolist(),
                "twist_resid": resids[off_t:off_t + twists.shape[1]].tolist(),
                "twist_mean": np.mean(twists, axis=0).tolist(),
            }
            # bend local de HELANAL, frame a frame, en cada residuo pedido
            bend_resids = np.asarray(helanal_out["bend_resid"])
            for site in kink_sites:
                r = site["resid"]
                j = np.where(bend_resids == r)[0]
                name = "bend_helanal_r%d_deg" % r
                if len(j) == 0:
                    warn("HELANAL no define bend local en el resid %d: el perfil solo "
                         "cubre %d-%d (los 3 primeros y 3 ultimos CA del rango no "
                         "tienen ventana a ambos lados). Se escribe NaN en %s; amplia "
                         "--resid-range si lo quieres."
                         % (r, bend_resids[0], bend_resids[-1], name))
                    col = np.full(data.shape[0], np.nan)
                elif bends.shape[0] != data.shape[0]:
                    warn("HELANAL analizo %d frames y el bucle principal %d; no se "
                         "pueden alinear los bends locales. Se escribe NaN en %s."
                         % (bends.shape[0], data.shape[0], name))
                    col = np.full(data.shape[0], np.nan)
                else:
                    col = bends[:, int(j[0])]
                data = np.column_stack([data, col])
                columns.append(name)
                helanal_out.setdefault("kink_sites", {})[str(r)] = {
                    "bend_mean": float(np.nanmean(col)),
                    "bend_std": float(np.nanstd(col)),
                }

            hinge = resids[off_b + int(np.argmax(np.mean(bends, axis=0)))]
            info("HELANAL: tilt global %.1f +- %.1f deg (plegado a [0,90]: %.1f; el signo "
                 "del eje global de HELANAL es ambiguo, por eso puede salir 180-x); "
                 "bisagra estimada (bend local maximo) cerca del resid %d."
                 % (helanal_out["global_tilt_mean"], helanal_out["global_tilt_std"],
                    helanal_out["global_tilt_folded_mean"], hinge))
            tw = np.mean(twists)
            if not (94 <= tw <= 106):
                warn("Giro local medio = %.1f deg/residuo (helice alfa ~ 100 deg). "
                     "Puede indicar deformacion, 3-10 o perdida de estructura." % tw)

    return u, data, columns, helanal_out, (n_rng, c_rng, kink_sites), box_lengths_0


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def write_outputs(args, data, columns, helanal_out, segments):
    prefix = args.out_prefix

    header = ("Generado por helix_kink_tilt.py el %s\n%s\n%s"
              % (datetime.now().isoformat(timespec="seconds"),
                 " ".join(sys.argv), ",".join(columns)))
    csv_path = "%s_timeseries.csv" % prefix
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="# ", fmt="%.6f")
    info("Escrito %s" % csv_path)

    def stats(col):
        v = data[:, columns.index(col)]
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                "median": float(np.nanmedian(v)), "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v)),
                "p5": float(np.nanpercentile(v, 5)), "p95": float(np.nanpercentile(v, 95))}

    kink_cols = [c for c in columns if c.startswith("kink_r")]
    bend_cols = [c for c in columns if c.startswith("bend_helanal_r")]
    kink_sites = segments[2] if len(segments) > 2 else []

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "versions": {"MDAnalysis": mda.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "arguments": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in vars(args).items()},
        "segments": {"nter": list(segments[0]), "cter": list(segments[1]),
                     "kink_sites": [{"resid": s["resid"], "window_nter": list(s["nter"]),
                                     "window_cter": list(s["cter"])}
                                    for s in kink_sites]},
        "n_frames_analyzed": int(data.shape[0]),
        "statistics": {c: stats(c) for c in
                       ["kink_deg", "tilt_full_deg", "tilt_full_folded_deg",
                        "tilt_nter_deg", "tilt_cter_deg", "rise_per_res_A"]
                       + kink_cols + bend_cols},
        "helanal": helanal_out,
        "checks": [{"level": lv, "message": m} for lv, m in CHECKS],
    }
    json_path = "%s_summary.json" % prefix
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    info("Escrito %s" % json_path)

    if helanal_out:
        hp = "%s_helanal_profile.csv" % prefix
        with open(hp, "w") as fh:
            fh.write("# resid,bend_mean_deg,bend_std_deg\n")
            for r, b, s in zip(helanal_out["bend_resid"], helanal_out["bend_mean"],
                               helanal_out["bend_std"]):
                fh.write("%d,%.4f,%.4f\n" % (r, b, s))
        info("Escrito %s" % hp)

    # Resumen en pantalla
    print("\n=== RESUMEN (%d frames) ===" % data.shape[0])
    for c in ["kink_deg", "tilt_full_deg", "tilt_nter_deg", "tilt_cter_deg"]:
        s = stats(c)
        print("  %-22s %6.2f +- %5.2f deg   (mediana %6.2f, rango %.2f-%.2f)"
              % (c, s["mean"], s["std"], s["median"], s["min"], s["max"]))

    if kink_cols:
        print("\n--- Kink por residuo bisagra ---")
        print("  %-8s %-24s %-22s %s"
              % ("resid", "ventanas (N vs C)", "kink ventanas (deg)", "bend HELANAL (deg)"))
        for site in kink_sites:
            r = site["resid"]
            kc = "kink_r%d_deg" % r
            bc = "bend_helanal_r%d_deg" % r
            sk = stats(kc)
            win = "%d-%d vs %d-%d" % (site["nter"][0], site["nter"][1],
                                      site["cter"][0], site["cter"][1])
            if bc in columns and not np.all(np.isnan(data[:, columns.index(bc)])):
                sb = stats(bc)
                btxt = "%6.2f +- %5.2f" % (sb["mean"], sb["std"])
            else:
                btxt = "n/d"
            print("  %-8d %-24s %6.2f +- %5.2f        %s"
                  % (r, win, sk["mean"], sk["std"], btxt))
    print()

    if args.no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warn("matplotlib no disponible: se omiten las graficas.")
        return

    t = data[:, columns.index("time_ps")] / 1000.0  # ns
    kink = data[:, columns.index("kink_deg")]
    tilt_full = data[:, columns.index("tilt_full_deg")]

    ncols = 3 if helanal_out else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 7))
    axes = np.atleast_2d(axes)

    axes[0, 0].plot(t, kink, lw=0.8, color="tab:red")
    axes[0, 0].set(xlabel="tiempo (ns)", ylabel="kink (deg)", title="Kink N-ter vs C-ter")
    axes[1, 0].hist(kink, bins=40, color="tab:red", alpha=0.8)
    axes[1, 0].set(xlabel="kink (deg)", ylabel="cuentas")

    axes[0, 1].plot(t, tilt_full, lw=0.8)
    axes[0, 1].set(xlabel="tiempo (ns)", ylabel="tilt (deg)",
                   title="Tilt de la helice completa vs normal de membrana")
    axes[1, 1].hist(tilt_full, bins=40, alpha=0.8)
    axes[1, 1].set(xlabel="tilt (deg)", ylabel="cuentas")

    if helanal_out:
        r = helanal_out["bend_resid"]
        b = np.array(helanal_out["bend_mean"])
        e = np.array(helanal_out["bend_std"])
        axes[0, 2].plot(r, b, "o-", ms=3, color="tab:green")
        axes[0, 2].fill_between(r, b - e, b + e, alpha=0.25, color="tab:green")
        for rk in (args.kink_resid or []):
            axes[0, 2].axvline(rk, ls=":", c="tab:red", lw=1)
        axes[0, 2].set(xlabel="resid", ylabel="bend local (deg)",
                       title="Perfil HELANAL (localiza la bisagra)")
        axes[1, 2].plot(helanal_out["twist_resid"], helanal_out["twist_mean"],
                        "o-", ms=3, color="tab:purple")
        axes[1, 2].axhline(100, ls="--", c="k", lw=0.8)
        axes[1, 2].set(xlabel="resid", ylabel="giro local (deg/res)", title="Helicidad")

    fig.tight_layout()
    png = "%s.png" % prefix
    fig.savefig(png, dpi=200)
    info("Escrito %s" % png)

    if not kink_cols:
        return

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    for site in kink_sites:
        r = site["resid"]
        v = data[:, columns.index("kink_r%d_deg" % r)]
        lbl = "resid %d" % r
        ax2[0].plot(t, v, lw=0.8, label=lbl)
        ax2[1].hist(v, bins=40, histtype="step", lw=1.4, label=lbl)
        bc = "bend_helanal_r%d_deg" % r
        if bc in columns:
            vb = data[:, columns.index(bc)]
            if not np.all(np.isnan(vb)):
                ax2[0].plot(t, vb, lw=0.6, alpha=0.45, ls="--",
                            label="resid %d (HELANAL)" % r)
    ax2[0].set(xlabel="tiempo (ns)", ylabel="angulo (deg)",
               title="Kink por residuo: ventanas (solido) y bend HELANAL (punteado)")
    ax2[1].set(xlabel="kink de ventanas (deg)", ylabel="cuentas", title="Distribucion")
    ax2[0].legend(fontsize=7, ncol=2)
    ax2[1].legend(fontsize=7)
    fig2.tight_layout()
    png2 = "%s_kinks.png" % prefix
    fig2.savefig(png2, dpi=200)
    info("Escrito %s" % png2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Kink y tilt de una helice transmembranal a lo largo de una trayectoria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Ejemplo")[-1])
    p.add_argument("-s", "--topology", required=True,
                   help="Topologia/estructura (tpr, gro, pdb, psf, parm7...). "
                        "El tpr/psf da nombres y conectividad correctos.")
    p.add_argument("-f", "--trajectory", nargs="*", default=[],
                   help="Trayectoria(s) (xtc, trr, dcd, nc...). Si se omite, se usa "
                        "solo la estructura (un frame).")
    p.add_argument("--resid-range", type=parse_range, required=True,
                   help="Rango de resid de la helice completa, p. ej. 700-723.")
    p.add_argument("--nter", type=parse_range, default=None,
                   help="Rango de resid del segmento N-terminal (p. ej. 700-707).")
    p.add_argument("--cter", type=parse_range, default=None,
                   help="Rango de resid del segmento C-terminal (p. ej. 712-723).")
    p.add_argument("--kink-resid", type=parse_resid_list, default=None,
                   metavar="17,21,25",
                   help="Residuos donde se hace el kink: calcula el angulo en cada uno "
                        "entre las ventanas de CA que lo flanquean, y el bend local de "
                        "HELANAL en el mismo residuo. Implica --helanal.")
    p.add_argument("--kink-window", type=int, default=7,
                   help="CA por ventana a cada lado de un residuo de --kink-resid "
                        "(def. 7; ~2 vueltas de helice). Acortala si las bisagras "
                        "pedidas estan a menos de --kink-window+--kink-gap+1 residuos "
                        "entre si, o las ventanas mezclaran varios codos.")
    p.add_argument("--kink-gap", type=int, default=1,
                   help="Residuos contiguos a la bisagra excluidos de las ventanas "
                        "(def. 1; el codo deforma la vuelta vecina).")
    p.add_argument("--gap", type=int, default=2,
                   help="Residuos excluidos en la bisagra al partir por defecto (def. 2).")
    p.add_argument("--min-seg", type=int, default=6,
                   help="Minimo de CA por segmento (def. 6; con menos, el eje es poco fiable).")
    p.add_argument("--select-extra", default="",
                   help="Selector extra MDAnalysis, p. ej. \"segid PROA\" o \"protein\".")
    p.add_argument("--ca-name", default="CA", help="Nombre del atomo de traza (def. CA).")
    p.add_argument("--lipid-sel", default="name P",
                   help="Atomos de referencia de la membrana (def. 'name P'; "
                        "en CHARMM tambien 'name P' funciona para POPC).")
    p.add_argument("--normal-mode", choices=["z", "leaflets", "plane"], default="leaflets",
                   help="Como estimar la normal de la membrana (def. leaflets).")
    p.add_argument("--axis-method", choices=["ca", "origins"], default="origins",
                   help="Ajuste del eje: SVD sobre CA, o sobre origenes locales HELANAL (def.).")
    p.add_argument("--linearity-cutoff", type=float, default=0.15,
                   help="Umbral s2/s1 para avisar de segmentos no rectos (def. 0.15).")
    p.add_argument("--helanal", action="store_true",
                   help="Correr HELANAL como control (tilt global, bends y giros locales).")
    p.add_argument("--first", type=int, default=None, help="Primer frame (0-based).")
    p.add_argument("--last", type=int, default=None, help="Ultimo frame (exclusivo).")
    p.add_argument("--stride", type=int, default=1, help="Paso entre frames.")
    p.add_argument("--no-make-whole", action="store_true",
                   help="No reconstruir la cadena rota por PBC (si ya usaste -pbc mol).")
    p.add_argument("--no-plots", action="store_true", help="No generar PNG.")
    p.add_argument("-o", "--out-prefix", default="kink_tilt", help="Prefijo de salida.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    u, data, columns, helanal_out, segments, _ = run_analysis(args)
    write_outputs(args, data, columns, helanal_out, segments)
    return 0


if __name__ == "__main__":
    sys.exit(main())