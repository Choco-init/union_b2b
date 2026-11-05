"""Generate selected or all examples; sends public data to OpenAI and incurs charges."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from product_assistant.data import parse_product
from product_assistant.generation import GenerationError, generate_card, result_json, result_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products', nargs='+', help='Product file stems, e.g. testo-608-h1 testo-608-h2')
    args = parser.parse_args()
    paths = sorted((ROOT / 'data/products').glob('*.json'))
    if args.products:
        unknown = set(args.products) - {path.stem for path in paths}
        if unknown:
            parser.error('Unknown products: ' + ', '.join(sorted(unknown)))
        paths = [path for path in paths if path.stem in args.products]
    output = ROOT / 'examples/generated'
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in paths:
        try:
            result = generate_card(parse_product(path.read_text(encoding='utf-8')))
        except GenerationError as error:
            print(f'{path.stem}: {error}', flush=True)
            return 1
        (output / (path.stem + '.json')).write_text(result_json(result), encoding='utf-8')
        (output / (path.stem + '.md')).write_text(result_markdown(result), encoding='utf-8')
        count = len(result['card']['seo_description']) if result['card'] else 0
        print(f'{path.stem}: {result["status"]}; description={count}; '
              f'attempts={result["attempts"]}; cost_usd={result["estimated_cost_usd"]}', flush=True)
        if result['status'] != 'ready':
            failures.append(path.stem)
            print('Validation:', result['validation_messages'], flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
