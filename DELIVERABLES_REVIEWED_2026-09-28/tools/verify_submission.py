"""Validate the mortality submission files, frozen data and executed notebook."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-executed', action='store_true', help='Also require saved error-free outputs in both notebooks')
    args = parser.parse_args()
    required = ['README.md','GAI_DECLARATION.md','AI_USE_LOG.md','SUBMISSION_CHECKLIST.md',
        'In_Hospital_Death.ipynb','requirements.txt','code/data.py','code/metrics.py',
        'models/L4_xgboost_best.json','models/L4_catboost_best.cbm',
        'preprocessing/mortality_preprocess.py','preprocessing/config.json',
        'preprocessing/mortality_data_preprocessing.ipynb','data/manifest.sha256']
    for name in required:
        p = ROOT / name
        assert p.is_file() and p.stat().st_size > 0, f'Missing or empty: {name}'
    readme = (ROOT/'README.md').read_text()
    words = len(readme.split())
    assert words <= 1500, f'README exceeds 1,500 words: {words}'
    assert 'Declaration on Use of GAI' in readme[:1000]
    print(f'PASS: {len(required)} required files; README {words}/1500 whitespace-delimited words')
    manifest_count = 0
    for line in (ROOT/'data/manifest.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        p = ROOT/'data'/name.strip()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, f'Changed snapshot: {name}'
        manifest_count += 1
    print(f'PASS: {manifest_count} frozen snapshot hashes')
    for folder in ['code','tools','tests','preprocessing']:
        for p in (ROOT/folder).rglob('*.py'):
            ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
    for name in ['In_Hospital_Death.ipynb','preprocessing/mortality_data_preprocessing.ipynb']:
        nb = nbformat.read(ROOT/name, as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        for c in code_cells:
            ast.parse(c.source)
            assert not any(o.output_type == 'error' for o in c.get('outputs',[])), f'Notebook error: {name}'
            if args.require_executed:
                assert c.get('execution_count') is not None, f'Unexecuted cell in {name}'
        print(f'PASS: {name}: {len(code_cells)} syntactically valid code cells' + (' with saved execution' if args.require_executed else ''))
    # Validate active relative documentation links (historical notes have old source links).
    for name in ['README.md','GAI_DECLARATION.md','AI_USE_LOG.md','SUBMISSION_CHECKLIST.md']:
        text = (ROOT/name).read_text()
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in link or link.startswith('#'):
                continue
            assert (ROOT/link.split('#')[0]).exists(), f'Broken link in {name}: {link}'
    print('PASS: active documentation links and Python syntax')
    report = ROOT/'artifacts/notebook/verification.json'
    if report.exists():
        result = json.loads(report.read_text())
        assert result['status'] == 'PASS'
        assert result['cohort'] == 11833 and result['features'] == 1474
        assert all(r['matches_reviewed'] for r in result['reviewed_metric_checks'])
        print('PASS: current notebook replay matches reviewed principal metrics')
    print('Submission checks complete (mortality component only).')


if __name__ == '__main__':
    main()
