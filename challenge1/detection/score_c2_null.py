"""Score the U-Net and the LOFO U-Nets on challenge-2 frames and empty scenes (report only).

    uv run python challenge1/detection/score_c2_null.py <dir of runs with best.pt + eval.json> challenge1/results/lofo/c2_null.json

False alarms = predicted blobs per stick-free scene, counted like metrics.object_scores
(fragments merged by a 3x3 dilation), the same counting as the classical baselines.
Threshold = each run's frozen val threshold (eval.json). Report-only sets."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["challenge1"]
from detection.evalsets import EvalSet, predict, score
from detection.models import build_model
from detection.export_dl import load

dev = torch.device("cpu")
torch.set_num_threads(8)
c2 = EvalSet("data/eval_light/ood_stage2")
null = EvalSet("data/eval_light/null")
CK = Path(sys.argv[1])

def run(model, thr, name):
    r2 = score(predict(model, c2.images, dev), c2, thr=thr)
    rn = score(predict(model, null.images, dev), null, thr=thr)
    out = {"name": name, "thr": thr, "c2_obj_f1": r2["obj_f1"], "c2_tol_f1": r2["tol_f1"],
           "null_blobs_per_img": rn["n_pred_blobs"] / len(null)}
    print(json.dumps(out), flush=True)
    return out

res = []
m, thr = load("challenge1/models/unet_lowsnr.pt")
res.append(run(m, thr, "unet_lowsnr"))
for d in sorted(CK.iterdir()):
    ck = torch.load(d / "best.pt", map_location=dev, weights_only=False)
    m = build_model(ck["arch"]); m.load_state_dict(ck["model"]); m.eval()
    thr = json.loads((d / "eval.json").read_text())["thr"]
    res.append(run(m, thr, d.name))
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
