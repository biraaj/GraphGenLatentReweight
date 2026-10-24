"""QM9 / ZINC250k generation metrics computed the DiGress / DeFoG way.

`build_molecule`, `build_molecule_with_partial_charges`, `mol2smiles`, `check_valency`,
`bond_dict`, `ATOM_VALENCY`, and the Validity / Relaxed Validity / Uniqueness / Novelty
logic are ported VERBATIM from DeFoG's `src/analysis/rdkit_functions.py`
(itself derived from DiGress) so the numbers are directly comparable to those papers.

FCD uses `fcd_torch` (the standard PyTorch reimplementation; MOSES uses it too) against
a reference SMILES set -- its ChemNet reference statistics are precomputed once and
cached, see `precompute_fcd_reference`.
"""
import os
import pickle
import re

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---- verbatim from DeFoG src/analysis/rdkit_functions.py --------------------------------
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    mol = Chem.RWMol()
    for atom in atom_types:
        mol.AddAtom(Chem.Atom(atom_decoder[atom.item()]))
    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0
    all_bonds = torch.nonzero(edge_types)
    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()])
    return mol


def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder, verbose=False):
    mol = Chem.RWMol()
    for atom in atom_types:
        mol.AddAtom(Chem.Atom(atom_decoder[atom.item()]))
    edge_types = torch.triu(edge_types)
    edge_types[edge_types >= 5] = 0
    all_bonds = torch.nonzero(edge_types)
    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()])
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            assert len(atomid_valence) == 2
            idx, v = atomid_valence[0], atomid_valence[1]
            an = mol.GetAtomWithIdx(idx).GetAtomicNum()
            if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


class BasicMolecularMetrics:
    """Validity / Relaxed Validity / Uniqueness / Novelty -- DeFoG's exact definitions."""

    def __init__(self, atom_decoder, train_smiles=None):
        self.atom_decoder = atom_decoder
        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        valid = []
        num_components = []
        for atom_types, edge_types in generated:
            mol = build_molecule(atom_types, edge_types, self.atom_decoder)
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except Exception:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    valid.append(mol2smiles(largest_mol))
                except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
                    pass
        return valid, len(valid) / len(generated), np.array(num_components)

    def compute_relaxed_validity(self, generated):
        valid = []
        for atom_types, edge_types in generated:
            mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    valid.append(mol2smiles(largest_mol))
                except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
                    pass
        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        if self.dataset_smiles_list is None:
            return unique, 1.0
        novel = [s for s in unique if s not in self.dataset_smiles_list]
        return novel, len(novel) / len(unique)

    def evaluate(self, generated):
        valid, validity, num_components = self.compute_validity(generated)
        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        nc_mu = float(num_components.mean()) if len(num_components) else 0.0
        nc_max = float(num_components.max()) if len(num_components) else 0.0
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            _, novelty = self.compute_novelty(unique)
        else:
            unique, uniqueness, novelty = [], 0.0, -1.0
        return {
            "Validity": validity,
            "RelaxedValidity": relaxed_validity,
            "Uniqueness": uniqueness,
            "Novelty": novelty,
            "nc_mu": nc_mu,
            "nc_max": nc_max,
            "_valid_smiles": valid,
            "_relaxed_valid_smiles": relaxed_valid,
            "_unique_smiles": unique,
        }


def precompute_fcd_reference(ref_smiles, cache_path, device="cpu"):
    """Precompute + cache the ChemNet reference statistics for FCD (mu, sigma)."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    from fcd_torch import FCD as FCDMetric

    fcd = FCDMetric(device=device, n_jobs=8, batch_size=512)
    pref = fcd.precalc([s for s in ref_smiles if s is not None])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(pref, f)
    return pref


def compute_fcd(gen_smiles, ref_smiles=None, ref_stats=None, device="cpu"):
    from fcd_torch import FCD as FCDMetric

    fcd = FCDMetric(device=device, n_jobs=8, batch_size=512)
    gen = [s for s in gen_smiles if s is not None]
    if not gen:
        return -1.0
    try:
        if ref_stats is not None:
            return float(fcd(gen=gen, pref=ref_stats))
        return float(fcd(gen=gen, ref=[s for s in ref_smiles if s is not None]))
    except Exception:
        return -1.0


def compute_digress_molecular_metrics(
    generated, atom_decoder, train_smiles=None, ref_smiles=None,
    fcd_ref_stats=None, device="cpu", dataset_name=None,
):
    """generated: list of (atom_types [n] long, edge_types [n,n] long, 0=no-bond, 1..=bonds).
    Returns the DiGress/DeFoG-style metric dict plus a combined VUN and FCD.
    """
    bmm = BasicMolecularMetrics(atom_decoder, train_smiles)
    res = bmm.evaluate(generated)

    n = len(generated)
    n_valid = len(res["_valid_smiles"])
    n_unique = len(set(res["_valid_smiles"]))
    n_vun = len({s for s in res["_valid_smiles"]
                 if train_smiles is None or s not in train_smiles})
    res["VU"] = n_unique / n if n else 0.0
    res["VUN"] = n_vun / n if n else 0.0

    gen_valid_smiles = res["_valid_smiles"]
    res["fcd"] = compute_fcd(gen_valid_smiles, ref_smiles=ref_smiles,
                             ref_stats=fcd_ref_stats, device=device)

    if dataset_name == "guacamol" and res["fcd"] is not None and res["fcd"] >= 0:
        # GuacaMol benchmark's own FCD metric (Brown et al. 2019) is this bounded
        # score, not the raw distance -- that's why published tables list Guacamol
        # FCD as "higher is better" while MOSES/QM9/ZINC tables list raw FCD as
        # "lower is better" for the same-named metric. Reported only for guacamol
        # so cross-dataset avg_ratio-style comparisons keep using the raw distance.
        import math
        res["fcd_score"] = math.exp(-0.2 * res["fcd"]) * 100.0

    return {k: v for k, v in res.items() if not k.startswith("_")}


# ── atom / molecule stability (DiGress / DeFoG, src/analysis/rdkit_functions.py) ────────
# Ported verbatim. Only meaningful when hydrogens are explicit (QM9 with-H); for
# heavy-atom-only datasets (ZINC250k, MOSES, guacamol) every generated atom is
# trivially "stable" so callers skip it — see stability_applies().

allowed_bonds = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Al": 3, "Si": 4,
    "P": [3, 5], "S": 4, "Cl": 1, "As": 3, "Br": 1, "I": 1, "Hg": [1, 2],
    "Bi": [3, 5], "Se": [2, 4, 6],
}


def stability_applies(atom_decoder) -> bool:
    """DiGress reports stability only for the explicit-H setting (QM9)."""
    return atom_decoder is not None and "H" in list(atom_decoder)


def check_stability(atom_types, edge_types, atom_decoder):
    """(molecule_stable, n_stable_bonds, n_atoms) — verbatim from DiGress/DeFoG."""
    n_bonds = np.zeros(len(atom_types), dtype="int")
    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            order = abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[i] += order
            n_bonds[j] += order
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible = allowed_bonds[atom_decoder[int(atom_type)]]
        is_stable = (possible == atom_n_bond) if isinstance(possible, int) else (atom_n_bond in possible)
        n_stable_bonds += int(is_stable)
    return n_stable_bonds == len(atom_types), n_stable_bonds, len(atom_types)


def compute_stability_metrics(generated, atom_decoder) -> dict:
    """{'mol_stable', 'atm_stable'} over the generated (atom_types, edge_types) tuples."""
    mol_stable = n_stable_bonds = n_atoms = 0
    for atom_types, edge_types in generated:
        s_mol, s_bonds, n = check_stability(atom_types, edge_types, atom_decoder)
        mol_stable += int(s_mol)
        n_stable_bonds += int(s_bonds)
        n_atoms += int(n)
    n_mol = max(len(generated), 1)
    return {
        "mol_stable": mol_stable / n_mol,
        "atm_stable": n_stable_bonds / max(n_atoms, 1),
    }


# ── NSPDK MMD over molecular graphs (GDSS/DruM convention) ─────────────────────────────

def mol_tuple_to_nx(atom_types, edge_types, atom_decoder):
    """(atom_types [n], edge_types [n,n]) -> nx.Graph with integer 'label' on nodes
    (atom-type index) and edges (bond order), the format eden.vectorize expects."""
    import networkx as nx

    g = nx.Graph()
    for i, a in enumerate(atom_types):
        g.add_node(i, label=int(a))
    n = len(atom_types)
    for i in range(n):
        for j in range(i + 1, n):
            b = int(edge_types[i, j])
            if 0 < b < 5:
                g.add_edge(i, j, label=b)
    # eden drops isolated nodes; keep the graph non-empty for the vectorizer
    if g.number_of_edges() == 0 and g.number_of_nodes() > 1:
        g.add_edge(0, 1, label=1)
    return g


def compute_nspdk(gen_tuples, ref_tuples, atom_decoder, n_jobs=8, max_ref=2000, max_gen=2000, seed=0):
    """NSPDK MMD between generated and reference molecular graphs. Lower = closer.

    Subsamples both sides to at most max_ref/max_gen graphs before vectorizing.
    eden's pairwise kernel is O(n^2) in memory: at MOSES's true reference-set
    scale (~176k test graphs) that's a ~176k x 176k float64 matrix, ~250 GB --
    the process gets SIGKILL'd by the OS OOM killer before any Python exception
    fires, so the try/except below can't catch it; the cap has to happen first.
    2000 is the scale GDSS/DruM-style NSPDK reports actually subsample to.
    """
    try:
        from evaluation.synthetic.dist_helper import compute_nspdk_mmd
        import random
        rng = random.Random(seed)
        ref_tuples = list(ref_tuples)
        gen_tuples = list(gen_tuples)
        if len(ref_tuples) > max_ref:
            ref_tuples = rng.sample(ref_tuples, max_ref)
        if len(gen_tuples) > max_gen:
            gen_tuples = rng.sample(gen_tuples, max_gen)
        gen_g = [mol_tuple_to_nx(a, e, atom_decoder) for a, e in gen_tuples]
        ref_g = [mol_tuple_to_nx(a, e, atom_decoder) for a, e in ref_tuples]
        gen_g = [g for g in gen_g if g.number_of_nodes() > 0]
        ref_g = [g for g in ref_g if g.number_of_nodes() > 0]
        if not gen_g or not ref_g:
            return -1.0
        return float(compute_nspdk_mmd(ref_g, gen_g, metric="nspdk", is_hist=False, n_jobs=n_jobs))
    except Exception as exc:  # eden/sklearn version issues shouldn't kill the sweep
        print(f"[nspdk] skipped: {exc}")
        return -1.0


def compute_molecular_metrics_full(
    generated, atom_decoder, train_smiles=None, ref_smiles=None,
    ref_tuples=None, fcd_ref_stats=None, device="cpu", nspdk=True, dataset_name=None,
):
    """compute_digress_molecular_metrics + molecule/atom stability + NSPDK MMD.

    generated / ref_tuples: lists of (atom_types [n] long, edge_types [n,n] long).
    Returns a flat dict of scalars: Validity, RelaxedValidity, Uniqueness, Novelty,
    VU, VUN, fcd, nc_mu, nc_max, [mol_stable, atm_stable], [nspdk], [fcd_score].
    """
    out = dict(compute_digress_molecular_metrics(
        generated, atom_decoder, train_smiles=train_smiles,
        ref_smiles=ref_smiles, fcd_ref_stats=fcd_ref_stats, device=device,
        dataset_name=dataset_name,
    ))
    if stability_applies(atom_decoder):
        out.update(compute_stability_metrics(generated, atom_decoder))
    if nspdk and ref_tuples:
        out["nspdk"] = compute_nspdk(generated, ref_tuples, atom_decoder)
    return out
