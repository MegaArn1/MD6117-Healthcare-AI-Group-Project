"""Execute the submitted notebook with the current Python interpreter."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parents[1] / 'In_Hospital_Death.ipynb')
    args = parser.parse_args()
    path = args.notebook.resolve()
    nb = nbformat.read(path, as_version=4)
    # A temporary kernelspec makes venv execution use this interpreter, not a global kernel.
    with tempfile.TemporaryDirectory(prefix='md6117-kernel-') as d:
        kernel = Path(d) / 'kernels' / 'md6117-submission'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'MD6117 current Python','language':'python'}))
        old = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = d + (os.pathsep + old if old else '')
        try:
            NotebookClient(nb, timeout=900, kernel_name='md6117-submission',
                resources={'metadata':{'path':str(path.parent)}}).execute()
        finally:
            if old is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = old
    nbformat.validate(nb)
    nbformat.write(nb, path)
    count = sum(c.cell_type == 'code' for c in nb.cells)
    print(f'PASS: executed {count} code cells in {path.name}')


if __name__ == '__main__':
    main()
