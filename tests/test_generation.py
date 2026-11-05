import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import AuthenticationError, RateLimitError, APITimeoutError

from product_assistant.data import parse_product, specification_rows
from product_assistant.generation import (
    Draft, GenerationError, generate_card, model_payload, validate_draft,
)

ROOT = Path(__file__).resolve().parents[1]


def product():
    return parse_product((ROOT / 'data/products/testo-0560-1109.json').read_text(encoding='utf-8'))


def valid_draft():
    return Draft(insufficient_data=False, reason='', seo_title='{{name}}',
                 seo_description='{{name}}. ' + ('Текст для проверки длины описания. ' * 28),
                 search_queries=['{{name}}', 'купить {{name}}', '{{name}} характеристики'],
                 commercial='Для специалистов, которым требуется измерять температуру поверхности.')


def response(draft=None, *, text=None, status='completed', refusal=False):
    return SimpleNamespace(
        output_text=text if text is not None else (draft or valid_draft()).model_dump_json(),
        status=status,
        output=[SimpleNamespace(content=[SimpleNamespace(type='refusal')])] if refusal else [],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                              input_tokens_details=SimpleNamespace(cached_tokens=0),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=50)),
    )


def fake_client(*responses):
    client = Mock()
    client.responses.create.side_effect = responses
    return client


class GenerationTests(unittest.TestCase):
    def test_conflicting_values_and_source_urls_do_not_enter_prompt(self):
        payload = json.dumps(model_payload(product()), ensure_ascii=False)
        self.assertNotIn('https://', payload)
        self.assertNotIn('±2%', payload)
        self.assertNotIn('±2 °C', payload)
        self.assertIn('Погрешность вне', payload)

    def test_safe_draft_copies_specs_and_usage(self):
        client = fake_client(response())
        result = generate_card(product(), client=client)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['card']['specifications'], specification_rows(product()))
        self.assertEqual(result['usage']['reasoning_tokens'], 50)
        self.assertEqual(result['estimated_cost_usd'], 0.0002)
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request['model'], 'gpt-6-luna')
        self.assertEqual(request['reasoning'], {'effort': 'low'})
        self.assertIs(request['store'], False)
        self.assertTrue(request['text']['format']['strict'])

    def test_technical_fact_substitution_preserves_conditions(self):
        draft = valid_draft()
        draft.commercial = 'Для подбора учитывайте: {{spec_2}}.'
        card, errors, evidence = validate_draft(draft, product())
        self.assertFalse(errors)
        self.assertIn('Погрешность при −30…+250 °C: ±1 °C', card['commercial'])
        self.assertTrue(any(e['input_field'] == '/specs/Погрешность при −30…+250 °C' for e in evidence))

    def test_new_number_and_unknown_reference_trigger_only_one_repair(self):
        bad = valid_draft()
        bad.commercial = 'Измеряет до 900 °C; {{spec_unknown}}.'
        client = fake_client(response(bad), response())
        result = generate_card(product(), client=client)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['usage']['input_tokens'], 2000)
        self.assertEqual(result['estimated_cost_usd'], 0.0004)
        self.assertIn('Замечания проверки', client.responses.create.call_args.kwargs['input'][-1]['content'])

    def test_persistent_invalid_output_is_not_a_publishable_card(self):
        client = fake_client(response(text='{bad'), response(text='{bad'))
        result = generate_card(product(), client=client)
        self.assertEqual(result['status'], 'needs_review')
        self.assertIsNone(result['card'])
        self.assertEqual(client.responses.create.call_count, 2)

    def test_title_and_description_limits_are_checked_after_substitution(self):
        draft = valid_draft()
        draft.seo_title = '{{name}} {{name}}'
        draft.seo_description = '{{name}}'
        _, errors, _ = validate_draft(draft, product())
        self.assertTrue(any('Заголовок' in e for e in errors))
        self.assertTrue(any('Описание' in e for e in errors))

    def test_1279_characters_shortened_without_another_api_request(self):
        draft = valid_draft()
        prefix = draft.seo_description.strip()
        prefix_length = len(validate_draft(draft, product())[0]['seo_description'])
        tail = ' Дополнительное пояснение ' + 'текст ' * 100
        draft.seo_description = prefix + tail[:1279 - prefix_length - 1] + '.'
        self.assertEqual(len(validate_draft(draft, product())[0]['seo_description']), 1279)
        client = fake_client(response(draft))
        result = generate_card(product(), client=client)
        self.assertEqual(result['status'], 'ready')
        description = result['card']['seo_description']
        self.assertTrue(800 <= len(description) <= 1200)
        self.assertTrue(description.endswith('.'))
        self.assertEqual(result['description_adjustment']['original_characters'], 1279)
        self.assertEqual(result['description_adjustment']['final_characters'], len(description))
        self.assertEqual(client.responses.create.call_count, 1)
        self.assertEqual(result['card']['specifications'], specification_rows(product()))

    def test_shortening_preserves_fact_with_decimal_and_conditions(self):
        draft = valid_draft()
        draft.seo_description = '{{spec_2}}. {{spec_4}}. ' + draft.seo_description
        draft.seo_description += ' Дополнительное пояснение ' + 'текст ' * 70 + '.'
        result = generate_card(product(), client=fake_client(response(draft)))
        self.assertEqual(result['status'], 'ready')
        description = result['card']['seo_description']
        self.assertIn('Погрешность при −30…+250 °C: ±1 °C', description)
        self.assertIn('Разрешение при −19,9…+199,9 °C: 0,1 °C', description)
        self.assertTrue(800 <= len(description) <= 1200)
        self.assertTrue(any(e['input_field'] == '/specs/Погрешность при −30…+250 °C'
                            for e in result['evidence']))

    def test_tail_with_facts_or_restrictions_is_not_silently_removed(self):
        for tail in ['{{spec_2}}.', 'Только при отсутствии конденсации.',
                     'Не использовать для других задач.', 'Погрешность зависит от условий.']:
            with self.subTest(tail=tail):
                draft = valid_draft()
                draft.seo_description += ' Дополнительное пояснение ' + 'текст ' * 80 + '. ' + tail
                client = fake_client(response(draft), response(draft))
                result = generate_card(product(), client=client)
                self.assertEqual(result['status'], 'needs_review')
                self.assertIsNone(result['card'])
                self.assertNotIn('description_adjustment', result)
                self.assertEqual(client.responses.create.call_count, 2)

    def test_shortening_does_not_hide_invalid_tail_or_publish_too_short_text(self):
        for description in [
            valid_draft().seo_description + ' ' + 'текст ' * 80 + ' 999 °C.',
            valid_draft().seo_description + ' ' + 'текст ' * 80 + ' {{unknown}}.',
            '{{name}}. ' + 'текст ' * 270 + '.',
        ]:
            with self.subTest(description=description[-30:]):
                draft = valid_draft()
                draft.seo_description = description
                result = generate_card(product(), client=fake_client(response(draft), response(draft)))
                self.assertEqual(result['status'], 'needs_review')
                self.assertIsNone(result['card'])

    def test_exact_1200_characters_are_not_shortened(self):
        draft = valid_draft()
        length = len(validate_draft(draft, product())[0]['seo_description'])
        draft.seo_description = draft.seo_description.strip() + ' ' + 'а' * (1200 - length - 2) + '.'
        result = generate_card(product(), client=fake_client(response(draft)))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(result['card']['seo_description']), 1200)
        self.assertNotIn('description_adjustment', result)

    def test_empty_input_never_calls_api(self):
        client = Mock()
        result = generate_card(parse_product('{"name":"X"}'), client=client)
        self.assertEqual(result['status'], 'insufficient_data')
        self.assertEqual(result['attempts'], 0)
        client.responses.create.assert_not_called()

    def test_refusal_and_incomplete_responses_do_not_trigger_repairs(self):
        for reply in [response(refusal=True), response(status='incomplete')]:
            with self.subTest(reply=reply.status):
                client = fake_client(reply)
                result = generate_card(product(), client=client)
                self.assertIsNone(result['card'])
                self.assertEqual(client.responses.create.call_count, 1)

    def test_api_errors_never_echo_sdk_message_or_key(self):
        request = httpx.Request('POST', 'https://api.openai.com/v1/responses')
        errors = [AuthenticationError('secret-must-not-leak', response=httpx.Response(401, request=request), body=None),
                  RateLimitError('secret-must-not-leak', response=httpx.Response(429, request=request), body=None),
                  APITimeoutError(request=request)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(GenerationError) as caught:
                    generate_card(product(), client=fake_client(error))
                self.assertNotIn('secret-must-not-leak', str(caught.exception))

    def test_missing_key_is_clear(self):
        with patch('product_assistant.generation.api_key', return_value=None):
            with self.assertRaisesRegex(GenerationError, 'OPENAI_API_KEY'):
                generate_card(product())


if __name__ == '__main__':
    unittest.main()
