#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import torch

DEFAULT_ILS = [
    {"candidate_id":"IL_001","candidate_name":"benzylimidazolium L-methioninate","anion_smiles":"CSCC[C@H](N)C(=O)[O-]","cation_smiles":"c1ccc(C[n+]2cc[nH]c2)cc1"},
    {"candidate_id":"IL_002","candidate_name":"EMIM 2,5-dihydroxybenzoate","anion_smiles":"O=C([O-])c1c(O)ccc(O)c1","cation_smiles":"CCn1cc[n+](C)c1"},
]

def loadmod(path, name):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def norm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan","none","null","na","n/a"} else s

def label(x): return re.sub(r"\s+", " ", norm(x))
def flist(s, low=-np.inf, high=None):
    vals=[]
    for p in str(s).split(','):
        if not p.strip(): continue
        v=float(p)
        if np.isfinite(v) and v>low and (high is None or v<=high): vals.append(round(v,8))
    if not vals: raise ValueError(f"No valid floats in {s}")
    return sorted(set(vals))

def charge_ok(train_mod, smi, q):
    mol=train_mod.mol_from_smiles(norm(smi))
    if mol is None: return False
    ch=train_mod.formal_charge(mol)
    return np.isfinite(ch) and int(round(ch))==q

def neutral_ok(train_mod, smi):
    smi=norm(smi)
    if not smi or '.' in smi: return False
    mol=train_mod.mol_from_smiles(smi)
    if mol is None: return False
    ch=train_mod.formal_charge(mol)
    return np.isfinite(ch) and int(round(ch))==0

def libraries(db, train_mod):
    df=train_mod.ensure_ion_smiles(db.copy())
    for p in ['IL1','IL2']:
        for c in [f'{p}_name',f'{p}_id',f'{p}_smiles',f'{p}_anion_smiles',f'{p}_cation_smiles',f'{p}_neutral_smiles']:
            if c not in df.columns: df[c]=''
    cats, ans, neuts, existing_pure, existing_pairs = {}, {}, {}, set(), set()
    for _,r in df.iterrows():
        comps={}
        for p in ['IL1','IL2']:
            cat, an, neu, whole = norm(r[f'{p}_cation_smiles']), norm(r[f'{p}_anion_smiles']), norm(r[f'{p}_neutral_smiles']), norm(r[f'{p}_smiles'])
            name, cid = label(r[f'{p}_name']), label(r[f'{p}_id'])
            if cat and charge_ok(train_mod, cat, +1):
                rec=cats.setdefault(cat, {'cation_smiles':cat,'cation_name':'','cation_id':'','n_seen':0}); rec['n_seen']+=1
                rec['cation_name']=rec['cation_name'] or name; rec['cation_id']=rec['cation_id'] or cid
            if an and charge_ok(train_mod, an, -1):
                rec=ans.setdefault(an, {'anion_smiles':an,'anion_name':'','anion_id':'','n_seen':0}); rec['n_seen']+=1
                rec['anion_name']=rec['anion_name'] or name; rec['anion_id']=rec['anion_id'] or cid
            ncan = neu or (whole if (not cat and not an and whole) else '')
            if ncan and neutral_ok(train_mod, ncan):
                rec=neuts.setdefault(ncan, {'neutral_smiles':ncan,'neutral_name':'','neutral_id':'','n_seen':0}); rec['n_seen']+=1
                rec['neutral_name']=rec['neutral_name'] or name; rec['neutral_id']=rec['neutral_id'] or cid
            if cat and an:
                key=f'{an}.{cat}'; existing_pure.add(key); comps[p]=('il',key)
            elif ncan and neutral_ok(train_mod, ncan): comps[p]=('neutral',ncan)
            else: comps[p]=('', '')
        k1,v1=comps.get('IL1',('','')); k2,v2=comps.get('IL2',('',''))
        if k1=='il' and k2=='neutral': existing_pairs.add(f'{v1}||{v2}')
        if k2=='il' and k1=='neutral': existing_pairs.add(f'{v2}||{v1}')
    def dfout(d, cols):
        if not d: return pd.DataFrame(columns=cols)
        return pd.DataFrame(list(d.values())).sort_values(['n_seen', cols[0]], ascending=[False, True]).reset_index(drop=True)
    return dfout(cats,['cation_smiles']), dfout(ans,['anion_smiles']), dfout(neuts,['neutral_smiles']), existing_pure, existing_pairs

def pure_candidates(cats, ans, temps, existing, max_combinations):
    if max_combinations>0 and len(cats)*len(ans)>max_combinations: raise ValueError('Too many pure combinations')
    rows=[]
    for _,c in cats.iterrows():
        for _,a in ans.iterrows():
            cat,an=norm(c.cation_smiles),norm(a.anion_smiles); il=f'{an}.{cat}'
            nm=f"{label(c.get('cation_name',''))} {label(a.get('anion_name',''))}".strip(); cid=f"{label(c.get('cation_id',''))}|{label(a.get('anion_id',''))}".strip('|')
            for T in temps:
                rows.append({'dataset_type':'pure','temperature_K':T,'x1':1.0,'x2':0.0,'IL1_name':nm,'IL1_id':cid,'IL1_smiles':il,'IL1_anion_smiles':an,'IL1_cation_smiles':cat,'IL1_neutral_smiles':'','IL2_name':'','IL2_id':'','IL2_smiles':'','IL2_anion_smiles':'','IL2_cation_smiles':'','IL2_neutral_smiles':'','generated_il_smiles':il,'is_existing_pure_il':int(il in existing)})
    return pd.DataFrame(rows)

def binary_candidates(cats, ans, neuts, temps, xs, existing_pairs, max_pairs, max_rows):
    n_pairs=len(cats)*len(ans)*len(neuts); n_rows=n_pairs*len(temps)*len(xs)
    if max_pairs>0 and n_pairs>max_pairs: raise ValueError(f'Too many IL-neutral pairs: {n_pairs:,}')
    if max_rows>0 and n_rows>max_rows: raise ValueError(f'Too many rows: {n_rows:,}')
    rows=[]
    for _,c in cats.iterrows():
        for _,a in ans.iterrows():
            cat,an=norm(c.cation_smiles),norm(a.anion_smiles); il=f'{an}.{cat}'
            nm=f"{label(c.get('cation_name',''))} {label(a.get('anion_name',''))}".strip(); cid=f"{label(c.get('cation_id',''))}|{label(a.get('anion_id',''))}".strip('|')
            for _,n in neuts.iterrows():
                neu=norm(n.neutral_smiles); nname=label(n.get('neutral_name','')); nid=label(n.get('neutral_id','')); pair=f'{il}||{neu}'
                for x in xs:
                    for T in temps:
                        rows.append({'dataset_type':'binary','temperature_K':T,'x1':float(x),'x2':float(1-x),'IL1_name':nm,'IL1_id':cid,'IL1_smiles':il,'IL1_anion_smiles':an,'IL1_cation_smiles':cat,'IL1_neutral_smiles':'','IL2_name':nname,'IL2_id':nid,'IL2_smiles':neu,'IL2_anion_smiles':'','IL2_cation_smiles':'','IL2_neutral_smiles':neu,'generated_il_smiles':il,'generated_neutral_smiles':neu,'generated_il_neutral_pair_key':pair,'is_existing_il_neutral_pair':int(pair in existing_pairs)})
    return pd.DataFrame(rows)

def selected_candidates(Tmin,Tmax,Tstep):
    temps=list(np.arange(float(Tmin), float(Tmax)+0.5*float(Tstep), float(Tstep)))
    if abs(temps[-1]-float(Tmax))>1e-9: temps.append(float(Tmax))
    rows=[]
    for d in DEFAULT_ILS:
        il=f"{d['anion_smiles']}.{d['cation_smiles']}"
        for T in temps:
            rows.append({'candidate_id':d['candidate_id'],'candidate_name':d['candidate_name'],'dataset_type':'pure','temperature_K':T,'x1':1.0,'x2':0.0,'IL1_name':d['candidate_name'],'IL1_id':d['candidate_id'],'IL1_smiles':il,'IL1_anion_smiles':d['anion_smiles'],'IL1_cation_smiles':d['cation_smiles'],'IL1_neutral_smiles':'','IL2_name':'','IL2_id':'','IL2_smiles':'','IL2_anion_smiles':'','IL2_cation_smiles':'','IL2_neutral_smiles':'','generated_il_smiles':il})
    return pd.DataFrame(rows)

def phase_pure(df, phase_model, phase_mod):
    obj=joblib.load(phase_model); model=obj['model']; cols=list(obj['feature_columns']); feats=[]
    for _,r in df.iterrows():
        s=pd.Series({'component_1_name':r.get('IL1_name',''),'component_1_id':r.get('IL1_id',''),'component_1_smiles':r.get('IL1_smiles',''),'component_1_anion_smiles':r.get('IL1_anion_smiles',''),'component_1_cation_smiles':r.get('IL1_cation_smiles',''),'component_1_primary_key':r.get('IL1_smiles',''),'component_2_name':'','component_2_id':'','component_2_smiles':'','component_2_anion_smiles':'','component_2_cation_smiles':'','component_2_primary_key':'','x1':1.0,'x2':0.0,'dataset_type':'pure'})
        feats.append(phase_mod.feature_block_from_components(phase_mod.component_block_from_row(s,'component_1'), phase_mod.zero_component_block(), 1.0,0.0,0.0))
    X=pd.DataFrame(feats)
    for c in cols:
        if c not in X.columns: X[c]=np.nan
    out=df.copy(); out['pred_phase_transition_K']=np.asarray(model.predict(X[cols].apply(pd.to_numeric,errors='coerce')),float); out['T_minus_pred_phase_K']=pd.to_numeric(out['temperature_K'],errors='coerce')-out['pred_phase_transition_K']; return out

def phase_binary(df, phase_model, phase_mod):
    obj=joblib.load(phase_model); model=obj['model']; cols=list(obj['feature_columns']); feats=[]
    for _,r in df.iterrows():
        s=pd.Series({'component_1_name':r.get('IL1_name',''),'component_1_id':r.get('IL1_id',''),'component_1_smiles':r.get('IL1_smiles',''),'component_1_anion_smiles':r.get('IL1_anion_smiles',''),'component_1_cation_smiles':r.get('IL1_cation_smiles',''),'component_1_neutral_smiles':r.get('IL1_neutral_smiles',''),'component_1_primary_key':r.get('IL1_smiles',''),'component_2_name':r.get('IL2_name',''),'component_2_id':r.get('IL2_id',''),'component_2_smiles':r.get('IL2_smiles',''),'component_2_anion_smiles':r.get('IL2_anion_smiles',''),'component_2_cation_smiles':r.get('IL2_cation_smiles',''),'component_2_neutral_smiles':r.get('IL2_neutral_smiles',''),'component_2_primary_key':r.get('IL2_smiles',''),'x1':r.get('x1',np.nan),'x2':r.get('x2',np.nan),'dataset_type':'binary'})
        kA,dA,kB,dB,xA,xB,xm=phase_mod.canonical_binary_components(s); feats.append(phase_mod.feature_block_from_components(dA,dB,xA,xB,xm))
    X=pd.DataFrame(feats)
    for c in cols:
        if c not in X.columns: X[c]=np.nan
    out=df.copy(); out['pred_phase_transition_K']=np.asarray(model.predict(X[cols].apply(pd.to_numeric,errors='coerce')),float); out['T_minus_pred_phase_K']=pd.to_numeric(out['temperature_K'],errors='coerce')-out['pred_phase_transition_K']; return out


def build_X(cand, train_mod, prep):
    point_df, group_df = train_mod.build_group_feature_table(train_mod.ensure_ion_smiles(cand.copy()))
    cols = list(prep["feature_cols"])
    for c in cols:
        if c not in group_df.columns:
            group_df[c] = np.nan
    X = prep["scaler"].transform(
        prep["imputer"].transform(group_df[cols].apply(pd.to_numeric, errors="coerce"))
    ).astype(np.float32)
    group_to_idx = {g: i for i, g in enumerate(group_df["group_id"].tolist())}
    gi = np.array([group_to_idx[g] for g in point_df["group_id"].values], dtype=np.int64)
    return point_df, X, gi


def load_multitarget_full_model(model_dir, mt_mod, device):
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "insitu_multitask_nn_no_morgan_FULL_clean_physics_loss_training.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing FULL physics-loss checkpoint: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    prep = joblib.load(model_dir / "preprocessing_and_prior.joblib")

    scaling = mt_mod.Scaling(**ck["scaling"])
    model = mt_mod.CleanPhysicsLossWrapper(
        n_features=len(ck["feature_cols"]),
        degree=int(ck["degree"]),
        hidden=list(ck["hidden"]),
        dropout=float(ck["dropout"]),
        scaling=scaling,
        cp_prior_params=dict(ck["cp_prior_params"]),
        rho_mean=float(ck.get("rho_mean", 1000.0)),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck, prep


def add_fom_columns(out, branch):
    if branch == "free":
        rho_col = "density_free_pred_kg_m3"
        cp_col = "cp_pred_JkgK"
        logeta_col = "log10_viscosity_free_pred_mPa_s"
        prefix = "free"
    elif branch == "struct":
        rho_col = "density_struct_pred_kg_m3"
        cp_col = "cp_pred_JkgK"
        logeta_col = "log10_viscosity_struct_pred_mPa_s"
        prefix = "struct"
    else:
        raise ValueError(branch)

    rho = pd.to_numeric(out[rho_col], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(out[cp_col], errors="coerce").to_numpy(float)
    logeta = pd.to_numeric(out[logeta_col], errors="coerce").to_numpy(float)
    mu = 10 ** np.clip(logeta, -12, 12)

    good = np.isfinite(rho) & np.isfinite(cp) & np.isfinite(mu) & (rho > 0) & (cp > 0) & (mu > 0)
    logf = np.full(len(out), np.nan, dtype=float)
    f = np.full(len(out), np.nan, dtype=float)
    logf[good] = 0.8*np.log10(rho[good]) + 0.4*np.log10(cp[good]) - 0.47*np.log10(mu[good])
    f[good] = 10 ** logf[good]

    out[f"viscosity_{prefix}_pred_mPa_s"] = mu
    out[f"log10_screening_merit_{prefix}"] = logf
    out[f"screening_merit_{prefix}_rho0p8_cp0p4_mu0p47"] = f
    out[f"physical_for_FOM_{prefix}"] = good.astype(int)
    return out


def predict_multitarget(cand, model_dir, train_mod, mt_mod, device, batch_size):
    model, ck, prep = load_multitarget_full_model(model_dir, mt_mod, device)
    p, X, gi = build_X(cand, train_mod, prep)

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    T = pd.to_numeric(p["temperature_K"], errors="coerce").to_numpy(np.float32)

    rho_free = np.empty(len(p), dtype=np.float32)
    rho_struct = np.empty(len(p), dtype=np.float32)
    cp_pred = np.empty(len(p), dtype=np.float32)
    logeta_free = np.empty(len(p), dtype=np.float32)
    logeta_struct = np.empty(len(p), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, len(p), batch_size):
            e = min(s + batch_size, len(p))
            group_idx = torch.tensor(gi[s:e], dtype=torch.long, device=device)
            tt = torch.tensor(T[s:e], dtype=torch.float32, device=device)
            oo = model(Xt, group_idx, tt)

            rho_free[s:e] = oo["rho_free_pred"].detach().cpu().numpy()
            rho_struct[s:e] = oo["rho_pred"].detach().cpu().numpy()
            cp_pred[s:e] = oo["cp_pred"].detach().cpu().numpy()
            logeta_free[s:e] = oo["logeta_free_pred"].detach().cpu().numpy()
            logeta_struct[s:e] = oo["logeta_pred"].detach().cpu().numpy()

    out = p.copy()
    out["density_free_pred_kg_m3"] = rho_free
    out["density_struct_pred_kg_m3"] = rho_struct
    out["cp_pred_JkgK"] = cp_pred
    out["log10_viscosity_free_pred_mPa_s"] = logeta_free
    out["log10_viscosity_struct_pred_mPa_s"] = logeta_struct

    # Compatibility/default prediction columns use the free branch.
    out["density_pred_kg_m3"] = out["density_free_pred_kg_m3"]
    out["log10_viscosity_pred_mPa_s"] = out["log10_viscosity_free_pred_mPa_s"]
    out["viscosity_pred_mPa_s"] = 10 ** np.clip(out["log10_viscosity_pred_mPa_s"].to_numpy(float), -12, 12)

    out = add_fom_columns(out, "free")
    out = add_fom_columns(out, "struct")

    # Legacy/default FOM columns use free branch.
    out["log10_screening_merit"] = out["log10_screening_merit_free"]
    out["screening_merit_rho0p8_cp0p4_mu0p47"] = out["screening_merit_free_rho0p8_cp0p4_mu0p47"]
    out["physical_for_FOM"] = out["physical_for_FOM_free"]

    # Consistency diagnostics.
    out["abs_density_free_minus_struct_kg_m3"] = np.abs(out["density_free_pred_kg_m3"] - out["density_struct_pred_kg_m3"])
    out["abs_logvisc_free_minus_struct"] = np.abs(out["log10_viscosity_free_pred_mPa_s"] - out["log10_viscosity_struct_pred_mPa_s"])
    return out


def write_ranked_outputs(df, outdir, kind, topn):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outs = {}

    def write_one_branch(branch):
        merit_col = f"screening_merit_{branch}_rho0p8_cp0p4_mu0p47"
        ranked = df.sort_values(merit_col, ascending=False)
        branch_tag = "free" if branch == "free" else "structured"

        if kind == "pure":
            groups = {
                "include_existing": ranked,
                "exclude_existing": ranked[ranked["is_existing_pure_il"].astype(int).eq(0)],
                "existing_only": ranked[ranked["is_existing_pure_il"].astype(int).eq(1)],
            }
            for tag, sub in groups.items():
                allp = outdir / f"all_pure_predictions_ranked_{tag}_{branch_tag}.csv"
                topp = outdir / f"top{topn}_{tag}_pure_by_{branch_tag}_merit.csv"
                sub.to_csv(allp, index=False)
                sub.head(topn).to_csv(topp, index=False)
                outs[f"top_{tag}_{branch_tag}"] = str(topp)

        elif kind == "binary":
            groups = {
                "include_existing": ranked,
                "exclude_existing": ranked[ranked["is_existing_il_neutral_pair"].astype(int).eq(0)],
                "existing_only": ranked[ranked["is_existing_il_neutral_pair"].astype(int).eq(1)],
            }
            for tag, sub in groups.items():
                allp = outdir / f"all_binary_il_neutral_predictions_ranked_{tag}_{branch_tag}.csv"
                topp = outdir / f"top{topn}_{tag}_binary_il_neutral_by_{branch_tag}_merit.csv"
                sub.to_csv(allp, index=False)
                sub.head(topn).to_csv(topp, index=False)
                outs[f"top_{tag}_{branch_tag}"] = str(topp)

    if kind in {"pure", "binary"}:
        write_one_branch("free")
        write_one_branch("struct")
        allp = outdir / f"all_{kind}_predictions_with_free_and_structured_branches.csv"
        df.to_csv(allp, index=False)
        outs["all_predictions_both_branches"] = str(allp)
    else:
        p = outdir / "selected_ils_multitarget_clean_physicsloss_noCpPrior_predictions_free_and_structured.csv"
        df.sort_values(["candidate_id", "temperature_K"]).to_csv(p, index=False)
        outs["selected_predictions"] = str(p)

    # Summary of free vs structured disagreement.
    summary = {
        "n_rows": int(len(df)),
        "mean_abs_density_free_minus_struct_kg_m3": float(pd.to_numeric(df["abs_density_free_minus_struct_kg_m3"], errors="coerce").mean()),
        "median_abs_density_free_minus_struct_kg_m3": float(pd.to_numeric(df["abs_density_free_minus_struct_kg_m3"], errors="coerce").median()),
        "mean_abs_logvisc_free_minus_struct": float(pd.to_numeric(df["abs_logvisc_free_minus_struct"], errors="coerce").mean()),
        "median_abs_logvisc_free_minus_struct": float(pd.to_numeric(df["abs_logvisc_free_minus_struct"], errors="coerce").median()),
    }
    pd.DataFrame([summary]).to_csv(outdir / "free_vs_structured_consistency_summary.csv", index=False)
    outs["free_vs_structured_consistency_summary"] = str(outdir / "free_vs_structured_consistency_summary.csv")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "binary", "selected"], required=True)
    ap.add_argument("--database", default="")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_train_script", default="train_insitu_multitask_nn_CLEAN_physics_loss_noCpPrior_Cpconst_noTphase_no_morgan.py")
    ap.add_argument("--feature_builder_script", default="train_insitu_multitask_nn_with_phase_filtered_no_morgan.py")
    ap.add_argument("--phase_feature_script", default="train_one_phase_transition_temperature_model_no_morgan.py")
    ap.add_argument("--phase_model", default="")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--temperatures", default="250")
    ap.add_argument("--x_il_grid", default="0.25,0.50,0.75")
    ap.add_argument("--phase_margin_K", type=float, default=10.0)
    ap.add_argument("--no_phase_filter", action="store_true")

    ap.add_argument("--T_min", type=float, default=150.0)
    ap.add_argument("--T_max", type=float, default=350.0)
    ap.add_argument("--T_step", type=float, default=5.0)

    ap.add_argument("--max_cations", type=int, default=50)
    ap.add_argument("--max_anions", type=int, default=30)
    ap.add_argument("--max_neutrals", type=int, default=15)
    ap.add_argument("--max_combinations", type=int, default=250000)
    ap.add_argument("--max_pairs", type=int, default=250000)
    ap.add_argument("--max_rows", type=int, default=1500000)

    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num_threads", type=int, default=1)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    train_mod = loadmod(args.feature_builder_script, "feature_builder_for_multitarget_noCpPrior_screen")
    mt_mod = loadmod(args.model_train_script, "mt_noCpPrior_model_for_screen")
    phase_mod = loadmod(args.phase_feature_script, "phase_model_features_for_screen") if args.phase_model else None

    if args.mode in {"pure", "binary"}:
        if not args.database:
            raise ValueError("--database is required for pure/binary mode.")
        temps = flist(args.temperatures, low=0.0)

        print(f"Reading database: {args.database}")
        db = pd.read_csv(args.database, low_memory=False)
        cats, ans, neuts, existing_pure, existing_pairs = libraries(db, train_mod)

        if args.mode == "pure":
            print(f"Unique cations: {len(cats)} | unique anions: {len(ans)}")
            cand = pure_candidates(cats, ans, temps, existing_pure, args.max_combinations)
            kind = "pure"
            cand.to_csv(Path(args.output_dir) / "generated_pure_inputs_before_phase_filter.csv", index=False)
            if args.phase_model:
                cand = phase_pure(cand, Path(args.phase_model), phase_mod)
                cand.to_csv(Path(args.output_dir) / "generated_pure_inputs_with_phase.csv", index=False)
                if not args.no_phase_filter:
                    cand = cand[pd.to_numeric(cand["T_minus_pred_phase_K"], errors="coerce") >= float(args.phase_margin_K)].copy()
                    cand.to_csv(Path(args.output_dir) / "generated_pure_inputs_phase_filtered.csv", index=False)

        else:
            xs = flist(args.x_il_grid, low=0.0, high=0.999999)
            cats = cats.head(args.max_cations if args.max_cations > 0 else len(cats)).reset_index(drop=True)
            ans = ans.head(args.max_anions if args.max_anions > 0 else len(ans)).reset_index(drop=True)
            neuts = neuts.head(args.max_neutrals if args.max_neutrals > 0 else len(neuts)).reset_index(drop=True)
            print(f"Using cations={len(cats)}, anions={len(ans)}, neutrals={len(neuts)}, x_grid={xs}, temperatures={temps}")
            cand = binary_candidates(cats, ans, neuts, temps, xs, existing_pairs, args.max_pairs, args.max_rows)
            kind = "binary"
            cand.to_csv(Path(args.output_dir) / "generated_binary_inputs_before_phase_filter.csv", index=False)
            if args.phase_model:
                cand = phase_binary(cand, Path(args.phase_model), phase_mod)
                cand.to_csv(Path(args.output_dir) / "generated_binary_inputs_with_phase.csv", index=False)
                if not args.no_phase_filter:
                    cand = cand[pd.to_numeric(cand["T_minus_pred_phase_K"], errors="coerce") >= float(args.phase_margin_K)].copy()
                    cand.to_csv(Path(args.output_dir) / "generated_binary_inputs_phase_filtered.csv", index=False)

    else:
        cand = selected_candidates(args.T_min, args.T_max, args.T_step)
        kind = "selected"
        cand.to_csv(Path(args.output_dir) / "selected_ils_inputs.csv", index=False)
        if args.phase_model:
            cand = phase_pure(cand, Path(args.phase_model), phase_mod)
            cand.to_csv(Path(args.output_dir) / "selected_ils_inputs_with_phase.csv", index=False)

    if cand.empty:
        raise ValueError("No candidates remain for prediction.")

    print(f"Predicting {len(cand):,} rows with FULL multitarget clean physics-loss no-Cp-prior model...")
    pred = predict_multitarget(cand, args.model_dir, train_mod, mt_mod, device, int(args.batch_size))
    outs = write_ranked_outputs(pred, args.output_dir, kind, int(args.top_n))

    with open(Path(args.output_dir) / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "model_dir": args.model_dir,
            "model_train_script": args.model_train_script,
            "primary_branch_for_default_columns": "free",
            "branches_written": ["free", "structured"],
            "n_rows_predicted": int(len(pred)),
            "outputs": outs,
        }, f, indent=2)

    print("Wrote outputs:")
    for k, v in outs.items():
        print(f"  {k}: {v}")
    print(f"  summary: {Path(args.output_dir) / 'run_summary.json'}")


if __name__ == "__main__":
    main()
