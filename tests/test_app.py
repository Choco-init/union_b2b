import json
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def start(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
        self.assertFalse(app.exception)
        return app

    def test_switch_preserves_edits_and_resets_only_selected_product(self):
        app = self.start()
        raw = json.loads(app.text_area(key='editor').value)
        raw['name'] = 'Мой черновик Testo'
        app.text_area(key='editor').set_value(json.dumps(raw)).run()
        app.selectbox(key='product_choice').select('APPA 61 · цифровой мультиметр').run()
        self.assertFalse(app.exception)
        self.assertIn('APPA', app.text_area(key='editor').value)
        app.selectbox(key='product_choice').select('Testo · поверхностный термометр').run()
        self.assertEqual(json.loads(app.text_area(key='editor').value)['name'], 'Мой черновик Testo')
        next(button for button in app.button if button.label.startswith('Сбросить')).click().run()
        self.assertNotEqual(json.loads(app.text_area(key='editor').value)['name'], 'Мой черновик Testo')
        self.assertFalse(app.exception)

    def test_invalid_editor_hides_preview_and_downloads(self):
        app = self.start()
        self.assertEqual(len(app.get('download_button')), 2)
        app.text_area(key='editor').set_value('{broken').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.get('download_button'))
        app.text_area(key='editor').set_value('{"name":"Прибор"}').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[1].value, '3')
        self.assertEqual(len(app.get('download_button')), 2)

    def test_both_fixtures_render_expected_issue_counts(self):
        app = self.start()
        self.assertEqual(app.metric[0].value, '9')
        self.assertEqual(app.metric[1].value, '2')
        app.selectbox(key='product_choice').select('APPA 61 · цифровой мультиметр').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '11')
        self.assertEqual(app.metric[1].value, '4')

    def test_additional_products_show_saved_cards_without_missing_data_warning(self):
        with patch('product_assistant.generation.generate_card') as generate:
            app = self.start()
            for label in ['Testo 608-H1 · термогигрометр',
                          'Testo 608-H2 · термогигрометр с сигнализацией']:
                app.selectbox(key='product_choice').select(label).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, '22')
                self.assertEqual(app.metric[1].value, '0')
                app.button(key='show_saved').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('download_button')), 4)
                self.assertFalse(app.warning)
            generate.assert_not_called()

    def test_generation_is_explicit_and_stale_result_disappears(self):
        from product_assistant.data import parse_product
        from product_assistant.generation import generate_card
        from test_generation import fake_client, response
        fixture = parse_product((ROOT / 'data/products/testo-0560-1109.json').read_text(encoding='utf-8'))
        result = generate_card(fixture, client=fake_client(response()))
        with patch('product_assistant.generation.generate_card', return_value=result) as generate, \
                patch('product_assistant.settings.api_key', return_value='test-only'):
            app = self.start()
            generate.assert_not_called()
            app.button(key='generate').click().run()
            self.assertFalse(app.exception)
            generate.assert_called_once()
            self.assertEqual(len(app.get('download_button')), 4)
            raw = json.loads(app.text_area(key='editor').value)
            raw['name'] = 'Изменённый товар'
            app.text_area(key='editor').set_value(json.dumps(raw)).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')), 2)
            generate.assert_called_once()

    def test_saved_example_needs_no_key_or_api_and_rejects_changed_input(self):
        with patch('product_assistant.generation.generate_card') as generate, \
                patch('product_assistant.settings.api_key', return_value=None):
            app = self.start()
            self.assertTrue(app.button(key='generate').disabled)
            app.button(key='show_saved').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')), 4)
            raw = json.loads(app.text_area(key='editor').value)
            raw['name'] = 'Другой прибор'
            app.text_area(key='editor').set_value(json.dumps(raw)).run()
            app.button(key='show_saved').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('download_button')), 2)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
