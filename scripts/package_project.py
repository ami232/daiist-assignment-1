"""Create a portable backup ZIP locally. Never uploads or submits."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / 'daiist-assignment-1-prepared.zip'
EXCLUDE = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', 'tmp'}
with ZipFile(DEST, 'w', compression=ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if path.is_file() and not any(part in EXCLUDE for part in relative.parts) and path.name != '.DS_Store':
            z.write(path, arcname=str(Path(ROOT.name) / relative))
print(f'Created {DEST} ({DEST.stat().st_size:,} bytes). Nothing uploaded to Blackboard.')
