#!/usr/bin/env python3
"""Extract lingua-spherica source files from compressed bundle.

Run: cd lingua-spherica && python3 setup_source.py

This extracts all 14 source files (10 Python modules, 2 demos, README, example output)
from a compressed+base64-encoded bundle. The bundle exists because the files total ~194KB
and were compressed to ~53KB for efficient transport.

After extraction, run the demo:
    python3 demo_e2e.py
"""
import json, base64, zlib, os, sys

# Check if already extracted
if os.path.exists('lingua_spherica/engine.py') and os.path.exists('demo_e2e.py'):
    print('Source files already extracted. Run: python3 demo_e2e.py')
    sys.exit(0)

DATA_FILE = 'source_bundle.b64'
if not os.path.exists(DATA_FILE):
    print(f'ERROR: {DATA_FILE} not found. Make sure you are in the lingua-spherica/ directory.')
    sys.exit(1)

with open(DATA_FILE) as f:
    encoded = f.read().strip()

bundle = json.loads(zlib.decompress(base64.b64decode(encoded)))
for path, content in bundle.items():
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f'  extracted: {path} ({len(content):,} bytes)')

print(f'\nDone: {len(bundle)} files extracted.')
print('Run the demo: python3 demo_e2e.py')
