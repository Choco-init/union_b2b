import json
import unittest
from pathlib import Path

from product_assistant.data import (
    InputError, collect_issues, export_json, export_markdown, parse_product,
)

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_examples_round_trip_without_changing_specs(self):
        for path in (ROOT / 'data/products').glob('*.json'):
            with self.subTest(path=path.name):
                raw = json.loads(path.read_text(encoding='utf-8'))
                product = parse_product(json.dumps(raw, ensure_ascii=False))
                exported = parse_product(export_json(product))
                self.assertEqual(exported.specs, raw['specs'])
                self.assertEqual(len(collect_issues(product)), len(raw['data_issues']))

    def test_incomplete_product_reports_missing_fields_and_survives_export(self):
        product = parse_product('{"name":"Прибор", "specs":{"Питание":" "}}')
        self.assertEqual(product.specs['Питание'], None)
        self.assertEqual({issue.field for issue in collect_issues(product)},
                         {'/manufacturer', '/description', '/specs/Питание'})
        self.assertEqual(len(collect_issues(parse_product(export_json(product)))), 3)

    def test_invalid_inputs_are_actionable(self):
        for text in ['{', '[]', 'null', '{"name":" "}', '{"name":"X","specs":{"V":3}}',
                     '{"name":"X","name":"Y"}', '{"name":"X","specs":{"V":NaN}}',
                     '{"name":"X","spces":{}}']:
            with self.subTest(text=text), self.assertRaises(InputError):
                parse_product(text)

    def test_stale_issue_cannot_silently_coexist_with_new_value(self):
        raw = json.loads((ROOT / 'data/products/testo-0560-1109.json').read_text(encoding='utf-8'))
        raw['specs']['Погрешность вне −30…+250 °C'] = '±2 °C'
        with self.assertRaisesRegex(InputError, 'замечание'):
            parse_product(json.dumps(raw))

    def test_unknown_issue_source_is_rejected(self):
        with self.assertRaisesRegex(InputError, 'Неизвестный источник'):
            parse_product(json.dumps({'name': 'X', 'data_issues': [
                {'field': '/manufacturer', 'kind': 'missing', 'message': 'Нет', 'source_ids': ['absent']}
            ]}))

    def test_markdown_preserves_missing_values_and_escapes_table_content(self):
        product = parse_product(json.dumps({'name': '<script>X</script>',
                                           'specs': {'A|B': 'line1\nline2', 'IP': None}}))
        markdown = export_markdown(product)
        self.assertIn('A\\|B', markdown)
        self.assertIn('line1<br>line2', markdown)
        self.assertNotIn('<script>', markdown)
        self.assertIn('Данных нет', markdown)
        self.assertIn('Требует уточнения', markdown)

    def test_json_pointer_escaping(self):
        product = parse_product('{"name":"X","specs":{"V/A~B":null}}')
        exported = parse_product(export_json(product))
        self.assertIn('/specs/V~1A~0B', {issue.field for issue in exported.data_issues})


if __name__ == '__main__':
    unittest.main()
