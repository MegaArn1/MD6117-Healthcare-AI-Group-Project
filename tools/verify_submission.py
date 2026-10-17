"""Validate the mortality submission files, frozen data and executed notebook."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def check_relative_links(text, parent, label):
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in link or link.startswith('#'):
            continue
        assert (parent/link.split('#')[0]).exists(), f'Broken link in {label}: {link}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-executed', action='store_true', help='Also require saved error-free outputs in both notebooks')
    args = parser.parse_args()
    required = ['README.md','FINAL_REPORT.md','docs/README.md','docs/DATA_README.md',
        'docs/PRESENTATION_MATERIALS.md','docs/REFERENCES.md','docs/REPRODUCIBILITY_CHECKS.md','docs/GAI_DECLARATION.md','docs/AI_USE_LOG.md','docs/SUBMISSION_CHECKLIST.md',
        'In_Hospital_Death.ipynb','requirements.txt','code/data.py','code/metrics.py',
        'models/L4_xgboost_best.json','models/L4_catboost_best.cbm',
        'preprocessing/mortality_preprocess.py','preprocessing/config.json',
        'preprocessing/mortality_data_preprocessing.ipynb','data/manifest.sha256',
        'results/README.md','results/figures/README.md','results/notebook/README.md',
        'results/notebook/verification.json','results/notebook/manifest.json',
        'models/README.md','models/notebook/model_metadata.json']
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
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                check_relative_links(cell.source, (ROOT/name).parent, name)
        print(f'PASS: {name}: {len(code_cells)} syntactically valid code cells' + (' with saved execution' if args.require_executed else ''))
    # Current guides use the new document archive; historical quoted layouts are preserved.
    documents = [ROOT/'README.md', ROOT/'FINAL_REPORT.md', *sorted((ROOT/'docs').glob('*.md')),
                 ROOT/'preprocessing/README.md', ROOT/'models/README.md',
                 *sorted((ROOT/'results').rglob('README.md'))]
    for document in documents:
        check_relative_links(document.read_text(), document.parent, document.relative_to(ROOT))
    print('PASS: active documentation links and Python syntax')
    manifest = json.loads((ROOT/'results/notebook/manifest.json').read_text())
    expected_files = {
        *[f'results/figures/notebook/{name}' for name in [
            'eda_measurement_coverage.png','eda_outcomes_training.png','roc_pr_test.png',
            'calibration_test.png','feature_count_validation.png']],
        *[f'results/notebook/tables/{name}' for name in [
            'gcs_statistic_development.csv','ladder_test.csv','icu_subgroups.csv','reviewed_metric_checks.csv']],
        'models/notebook/L3_pipeline.joblib','models/notebook/L3-revised_pipeline.joblib',
        'models/notebook/model_metadata.json','results/notebook/verification.json',
    }
    assert set(manifest['files']) == expected_files, 'Published output inventory mismatch'
    for relative, metadata in manifest['files'].items():
        artifact = ROOT/relative
        assert artifact.stat().st_size == metadata['bytes'], f'Output size mismatch: {relative}'
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata['sha256'], f'Output hash mismatch: {relative}'
    print(f'PASS: {len(expected_files)} published figure/table/model output hashes')
    result = json.loads((ROOT/'results/notebook/verification.json').read_text())
    assert result['status'] == 'PASS'
    assert result['cohort'] == 11833 and result['features'] == 1474
    assert all(r['matches_reviewed'] for r in result['reviewed_metric_checks'])
    assert set(result['published_output_paths']) == expected_files
    def read_table(filename):
        with (ROOT/'results/notebook/tables'/filename).open(newline='') as handle:
            return list(csv.DictReader(handle))
    ladder = read_table('ladder_test.csv')
    assert len(ladder) == 9
    l4 = next(row for row in ladder if row['model'] == 'L4 XGBoost')
    assert int(l4['n']) == result['test_n']
    for column, metric in [('AUROC','auroc'),('AUPRC','auprc'),('sensitivity','sensitivity'),
                           ('ppv','ppv'),('specificity','specificity')]:
        assert math.isclose(float(l4[column]), result['l4_test_metrics'][metric], abs_tol=1e-12), column
    groups = read_table('icu_subgroups.csv')
    assert len(groups) == 4 and sum(int(row['n']) for row in groups) == result['test_n']
    assert sum(int(row['deaths']) for row in groups) == 256
    comparisons = read_table('reviewed_metric_checks.csv')
    assert len(comparisons) == 7 and all(row['matches_reviewed'] == 'True' for row in comparisons)
    assert len(read_table('gcs_statistic_development.csv')) == 7
    model_metadata = json.loads((ROOT/'models/notebook/model_metadata.json').read_text())
    assert model_metadata['environment'] == result['environment']
    assert set(model_metadata['models']) == {'L3','L3-revised'}
    for model in model_metadata['models'].values():
        assert len(model['features']) == 8 and model['validation_predictions_match_after_reload']
    print('PASS: published CSVs and model metadata agree with the successful notebook replay')
    print('Submission checks complete (mortality component only).')


if __name__ == '__main__':
    main()
