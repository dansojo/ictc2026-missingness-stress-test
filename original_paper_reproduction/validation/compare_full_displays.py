"""Compare newly rendered paper outputs with the submitted figure/table files."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper'))
from full_prepare import sha256
from full_g2 import load_stage
from PIL import Image
from pypdf import PdfReader


def compare(actual: Path, reference: Path) -> dict:
    manifest = load_stage(actual, 'displays')
    result = {'schema_name': 'fresh_paper_display_comparison_v1',
              'fresh_display_manifest_sha256': sha256(actual / 'manifest.json'), 'files': {}}
    for name in ['fig2_results.png', 'fig2_results.pdf',
                 'table1_feature_contracts.tex', 'table2_external_transfer.tex']:
        generated = actual / manifest['outputs'][name]['path']
        original = reference / name
        row = {'actual_sha256': sha256(generated), 'reference_sha256': sha256(original)}
        row['byte_identical'] = row['actual_sha256'] == row['reference_sha256']
        if generated.suffix == '.png':
            with Image.open(generated) as a, Image.open(original) as b:
                row['comparison'] = 'size and decoded RGBA pixels'
                row['content_equal'] = a.size == b.size and a.convert('RGBA').tobytes() == b.convert('RGBA').tobytes()
                row['actual_size'] = list(a.size)
        elif generated.suffix == '.tex':
            row['comparison'] = 'UTF-8 text, normalized line endings'
            row['content_equal'] = generated.read_text(encoding='utf-8-sig') == original.read_text(encoding='utf-8-sig')
        else:
            a, b = PdfReader(generated), PdfReader(original)
            row['comparison'] = 'page count and decoded page content streams'
            row['content_equal'] = len(a.pages) == len(b.pages) and all(
                x.get_contents().get_data() == y.get_contents().get_data() for x, y in zip(a.pages, b.pages))
            row['actual_pages'] = len(a.pages)
        result['files'][name] = row
    result['all_content_equal'] = all(row['content_equal'] for row in result['files'].values())
    result['status'] = 'complete' if result['all_content_equal'] else 'comparison_failed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'paper/reference/displays')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.actual_dir, args.reference_dir)
    with args.report.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'all_content_equal': report['all_content_equal']}))
    raise SystemExit(0 if report['all_content_equal'] else 1)
