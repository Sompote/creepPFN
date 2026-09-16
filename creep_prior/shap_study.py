"""Exact four-group interventional SHAP for frozen source-disjoint forecasts.

All 16 coalitions are enumerated. The empirical background has one eligible
curve per training source, repeated with an independently drawn second panel.
History includes times, responses and anchor descriptor. Modulus includes its
missingness flag. Query horizons remain fixed; later observed responses are
never used in the attribution calculation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .model import CreepPFN
from .train import real_records, pack, tensors, forward
from .pipeline import sha256

GROUPS = ['density', 'strength', 'modulus_and_missingness', 'early_history_and_anchor']
HORIZONS = [28.,90.,160.]


def exact_values(values):
    """values has shape (16, ...); return four exact Shapley contributions."""
    out = np.zeros((4,)+values.shape[1:], dtype=np.float64)
    for j in range(4):
        for mask in range(16):
            if mask & (1 << j):
                continue
            k=mask.bit_count()
            weight=math.factorial(k)*math.factorial(3-k)/math.factorial(4)
            out[j] += weight*(values[mask | (1<<j)]-values[mask])
    return out


def hybrid(target, donor, mask):
    x=np.array(donor['features'],copy=True)
    if mask & 1: x[0]=target['features'][0]
    if mask & 2: x[1]=target['features'][1]
    if mask & 4: x[2:4]=target['features'][2:4]
    history=target if mask & 8 else donor
    x[4]=history['features'][4]
    return dict(context_times=history['context_times'],context_values=history['context_values'],
                features=x,query_times=np.asarray(HORIZONS),targets=np.zeros(3))


def predict_ensemble(models, records, device, batch_size):
    outputs=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch_size):
            batch=tensors(pack(records[start:start+batch_size]),device)
            vals=[]
            for model in models:
                mu,_,scale=forward(model,batch)
                vals.append(torch.sinh(mu)*scale[:,None])
            outputs.append(torch.stack(vals).mean(0).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    root=args.root;out=root/'artifacts/shap_study';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(20260916)
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    start=time.time();manifest={};donor_manifest=[];all_rows=[]
    for fi in range(1,6):
        fold=f'fold_{fi:02}'
        prior_dir=root/'artifacts/cv5_priors'/fold
        training,_=real_records(prior_dir,'train')
        targets,_=real_records(prior_dir,'test')
        targets=sorted(targets,key=lambda r:r['curve_id'])
        if args.limit: targets=targets[:args.limit]
        training_groups=sorted(set(r['group'] for r in training))
        assert not set(r['group'] for r in targets)&set(training_groups)
        panels=[]
        for repeat in range(2):
            rng=np.random.default_rng(20260916+100*fi+repeat)
            panel=[]
            for group in training_groups:
                candidates=sorted([r for r in training if r['group']==group],key=lambda r:r['curve_id'])
                donor=candidates[int(rng.integers(len(candidates)))];panel.append(donor)
                donor_manifest.append(dict(fold=fold,panel=repeat,group=group,curve_id=donor['curve_id']))
            panels.append(panel)
        donors=panels[0]+panels[1];n=len(panels[0])
        models=[];checkpoint_hashes={}
        for seed in (45,46,47):
            path=root/f'artifacts/revised_main/{fold}/revised_main/seed{seed}/finetune/best.pt'
            weights=torch.load(path,map_location='cpu',weights_only=True)
            assert weights['training_config']['prior_sha256']==sha256(prior_dir/'prior.json')
            model=CreepPFN(**weights['model_config']).to(device)
            model.load_state_dict(weights['state_dict']);model.eval();models.append(model)
            checkpoint_hashes[str(seed)]=sha256(path)
        coalition_values=np.zeros((len(targets),2,16,3))
        for ti,target in enumerate(targets):
            records=[hybrid(target,d,mask) for mask in range(16) for d in donors]
            pred=predict_ensemble(models,records,device,args.batch_size).reshape(16,2,n,3)
            coalition_values[ti]=pred.mean(axis=2).transpose(1,0,2)
            if (ti+1)%20==0:print(f'{fold} {ti+1}/{len(targets)} elapsed={time.time()-start:.1f}s',flush=True)
        np.savez_compressed(out/f'{fold}_coalitions.npz',values=coalition_values,
                            curve_ids=np.array([r['curve_id'] for r in targets]))
        direct=predict_ensemble(models,[hybrid(r,r,15) for r in targets],device,args.batch_size)
        direct_error=float(np.max(np.abs(coalition_values[:,:,15,:]-direct[:,None,:])))
        if not np.allclose(coalition_values[:,:,15,:],direct[:,None,:],rtol=2e-5,atol=2e-4):
            raise ValueError(f'Unmasked prediction mismatch: {direct_error}')
        for ti,target in enumerate(targets):
            for repeat in range(2):
                values=coalition_values[ti,repeat];phi=exact_values(values)
                residual=values[15]-values[0]-phi.sum(axis=0)
                assert np.max(np.abs(residual))<1e-8
                for hi,horizon in enumerate(HORIZONS):
                    row=dict(fold=fold,curve_id=target['curve_id'],source=target['group'],
                             panel=repeat,horizon=horizon,base=values[0,hi],prediction=values[15,hi],
                             additivity_residual=residual[hi])
                    row.update({name:phi[j,hi] for j,name in enumerate(GROUPS)})
                    all_rows.append(row)
        manifest[fold]=dict(test_curves=len(targets),training_sources=n,checkpoint_hashes=checkpoint_hashes,
                            direct_prediction_max_abs_difference=direct_error,
                            prior_sha256=sha256(prior_dir/'prior.json'))
        pd.DataFrame(all_rows).to_csv(out/'attributions.csv',index=False)
        pd.DataFrame(donor_manifest).to_csv(out/'background.csv',index=False)
        print(f'{fold} complete',flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(groups=GROUPS,horizons=HORIZONS,
        background='Two panels, each drawing one eligible curve uniformly within each training source; equal source weights.',
        explained_output='Mean of three response medians, microstrain/MPa; fixed horizons; no observed future values.',
        interpretation='Exact coalition enumeration for an empirical interventional background; not causal or conditional SHAP.',
        folds=manifest,elapsed_seconds=time.time()-start,torch_version=torch.__version__,
        numpy_version=np.__version__,device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        script_sha256=sha256(__file__)),indent=2)+'\n')
    (out/'COMPLETE').write_text('Completed successfully\n')


if __name__=='__main__':main()
