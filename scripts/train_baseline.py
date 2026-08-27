import os
import sys
import glob
import yaml
import torch
import ultralytics
from ultralytics import YOLO

BASE = r'c:\Users\ksr20\OneDrive\Desktop\Lastmile Robotics'
DATA_YAML = os.path.join(BASE, 'dataset_button', 'data.yaml')
PROJECT_DIR = os.path.join(BASE, 'runs', 'detect')
RUN_NAME = 'button_yolov8n_fast_baseline'

EPOCHS = 30
IMGSZ = 512
BATCH = 8
PATIENCE = 8
SEED = 42
DEVICE = 'cpu'
WORKERS = 0

print('=== Effective Configuration ===')
print(f'epochs={EPOCHS}')
print(f'imgsz={IMGSZ}')
print(f'batch={BATCH}')
print(f'patience={PATIENCE}')
print(f'device={DEVICE}')
print(f'workers={WORKERS}')
print(f'seed={SEED}')
print(f'run_name={RUN_NAME}')
print(f'data={DATA_YAML}')

print('\n=== Environment ===')
print('python=' + sys.version.split()[0])
print('torch=' + torch.__version__)
print('cuda=' + str(torch.cuda.is_available()))
print('ultralytics=' + ultralytics.__version__)

print('\n=== Dataset Verification ===')
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)
print('nc=' + str(cfg['nc']) + ' names=' + str(cfg['names']))
ds_root = os.path.dirname(DATA_YAML)
for split, key in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
    img_dir = os.path.join(ds_root, cfg[key])
    lbl_dir = img_dir.replace('images', 'labels')
    imgs = glob.glob(os.path.join(img_dir, '*'))
    lbls = glob.glob(os.path.join(lbl_dir, '*.txt'))
    print(split + ': images=' + str(len(imgs)) + ' labels=' + str(len(lbls)))

print('\n=== Training ===')
model = YOLO('yolov8n.pt')

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    seed=SEED,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    verbose=True,
)

best_weights = os.path.join(PROJECT_DIR, RUN_NAME, 'weights', 'best.pt')

print('\n=== Validation (val split) ===')
model_best = YOLO(best_weights)
val_metrics = model_best.val(
    data=DATA_YAML,
    split='val',
    imgsz=IMGSZ,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT_DIR,
    name=RUN_NAME + '_val',
    exist_ok=True,
)
print('val/precision=' + str(round(float(val_metrics.box.mp), 4)))
print('val/recall=' + str(round(float(val_metrics.box.mr), 4)))
print('val/mAP50=' + str(round(float(val_metrics.box.map50), 4)))
print('val/mAP50-95=' + str(round(float(val_metrics.box.map), 4)))

print('\n=== Test (test split) ===')
test_metrics = model_best.val(
    data=DATA_YAML,
    split='test',
    imgsz=IMGSZ,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT_DIR,
    name=RUN_NAME + '_test',
    exist_ok=True,
)
print('test/precision=' + str(round(float(test_metrics.box.mp), 4)))
print('test/recall=' + str(round(float(test_metrics.box.mr), 4)))
print('test/mAP50=' + str(round(float(test_metrics.box.map50), 4)))
print('test/mAP50-95=' + str(round(float(test_metrics.box.map), 4)))

run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
print('\n=== Output Files ===')
print('best.pt:            ' + os.path.join(run_dir, 'weights', 'best.pt'))
print('last.pt:            ' + os.path.join(run_dir, 'weights', 'last.pt'))
print('results.csv:        ' + os.path.join(run_dir, 'results.csv'))
print('results.png:        ' + os.path.join(run_dir, 'results.png'))
print('confusion_matrix:   ' + os.path.join(run_dir, 'confusion_matrix.png'))
print('PR_curve:           ' + os.path.join(run_dir, 'PR_curve.png'))
print('F1_curve:           ' + os.path.join(run_dir, 'F1_curve.png'))
