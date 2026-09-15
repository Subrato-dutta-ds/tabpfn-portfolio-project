import os, json, subprocess
from datetime import datetime
from src.config import BASE_DIR

try:
    git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=BASE_DIR, stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_sha = 'unknown'

meta_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
with open(meta_path) as f:
    meta = json.load(f)

meta['git_sha'] = git_sha
meta['trained_at'] = datetime.utcnow().isoformat() + 'Z'
meta['feature_schema_version'] = '1.0'

with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=4)

print(f'Metadata updated: git_sha={git_sha}')
