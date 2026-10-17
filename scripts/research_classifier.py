import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.engine import LocalAnalyzer, read_frames, video_info
from scripts.evaluate import labels_from_xlsx, normalize

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, required=True)
args = parser.parse_args()
out = ROOT / 'data' / 'classifier-research'
out.mkdir(exist_ok=True)
labels = labels_from_xlsx(args.dataset / 'open_labels.xlsx')
videos = {normalize(p.stem): p for p in (args.dataset / 'videos').glob('*.mov')}
analyzer = LocalAnalyzer()
analyzer.load()
import torch
items, features, seen = [], [], set()
for label in labels:
    path = videos[normalize(label['name'])]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen:
        continue
    seen.add(digest)
    feature_path = out / (digest + '.npy')
    began = time.monotonic()
    if feature_path.exists():
        feature = np.load(feature_path, allow_pickle=False)
    else:
        info = video_info(path)
        frames = read_frames(path, np.linspace(0, info['duration_sec'] - .1, 16), 448)
        pooled = []
        for start in range(0, len(frames), 4):
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for _, frame in frames[start:start + 4]]
            inputs = analyzer.processor.image_processor(images=images, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                embeds, deep = analyzer.model.get_image_features(inputs.pixel_values, inputs.image_grid_thw)
                pooled.extend(e.float().mean(dim=0).cpu().numpy() for e in embeds)
            del inputs, embeds, deep
        matrix = np.stack(pooled)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        feature = np.concatenate([matrix.mean(0), matrix.std(0), matrix.max(0), np.abs(np.diff(matrix, axis=0)).mean(0)])
        np.save(feature_path, feature, allow_pickle=False)
    features.append(feature)
    items.append({'name':label['name'], 'sha256':digest, 'label':int(label['violation'])})
    print(json.dumps({'features_completed':len(items),'unique_total':40,'elapsed_sec':round(time.monotonic()-began,2)}),flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

X = np.stack(features)
y = np.array([i['label'] for i in items])
classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.01,class_weight='balanced',max_iter=1000,random_state=2026))
folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
probabilities = cross_val_predict(classifier,X,y,cv=folds,method='predict_proba')[:,1]
predicted = probabilities >= .5
report = {'method':'5-fold stratified cross-validation on 40 SHA-256-deduplicated training videos; fixed C=.01; preprocessing fitted only on each training fold',
          'unique_videos':len(items),'feature_dimensions':int(X.shape[1]),'accuracy':float(accuracy_score(y,predicted)),
          'precision':float(precision_score(y,predicted,zero_division=0)),'recall':float(recall_score(y,predicted,zero_division=0)),
          'confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,predicted).tolist(), 'independent_camera_test':False}
records = [dict(item,oof_probability=float(p),oof_predicted=int(v)) for item,p,v in zip(items,probabilities,predicted)]
(out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
(out/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report),flush=True)
