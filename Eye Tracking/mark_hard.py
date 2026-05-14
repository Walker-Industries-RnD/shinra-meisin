"""Stamp hard=True on the original captures/ JSONs that were mined into hard_mines/.

Clears all previous hard markings across captures/ first so there is no
carryover from prior runs.
"""

import json
import sys
from pathlib import Path

from input import REAL_DIR

HARD_DIR = sys.argv[1] if len(sys.argv) > 1 else 'hard_mines'


def _clear_all(captures_dir: Path) -> int:
    cleared = 0
    for json_path in captures_dir.rglob('*.json'):
        with open(json_path) as f:
            data = json.load(f)
        if data.pop('hard', None) is not None:
            with open(json_path, 'w') as f:
                json.dump(data, f)
            cleared += 1
    return cleared


def main():
    hard_dir = Path(HARD_DIR)
    if not hard_dir.exists():
        raise SystemExit(f'{hard_dir} not found')

    captures_dir = Path(REAL_DIR)
    cleared = _clear_all(captures_dir)
    print(f'Cleared {cleared} previous hard markings from {captures_dir}')

    patched = missing = 0
    for json_path in sorted(hard_dir.rglob('*.json')):
        rel = json_path.relative_to(hard_dir)
        orig = captures_dir / rel

        if not orig.exists():
            print(f'MISSING  {orig}')
            missing += 1
            continue

        with open(orig) as f:
            data = json.load(f)
        data['hard'] = True
        with open(orig, 'w') as f:
            json.dump(data, f)
        patched += 1

    print(f'Marked {patched} hard  |  missing {missing}')


if __name__ == '__main__':
    main()
