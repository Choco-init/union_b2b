"""Generate bounded, traceable drafts with GPT-6 Luna and validate locally."""
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal

from openai import (
    OpenAI, APIConnectionError, APIStatusError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, RateLimitError,
)
from pydantic import ValidationError

from product_assistant.data import (
    Product, StrictModel, collect_issues, escape_markdown, field_label,
    pointer_key, specification_rows,
)
from product_assistant.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from product_assistant.settings import api_key
from product_assistant.text_limits import DESCRIPTION_MIN, DESCRIPTION_MAX, shorten_description

MODEL = 'gpt-6-luna'
REASONING_EFFORT = 'low'
MAX_OUTPUT_TOKENS = 3500
MAX_PAYLOAD_CHARS = 32000
REFERENCE = re.compile(r'\{\{([a-zA-Z0-9_]+)\}\}')


class GenerationError(ValueError):
    """Safe public error: never include SDK exception strings or credentials."""


class Draft(StrictModel):
    insufficient_data: bool
    reason: str
    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial: str


def fingerprint(product: Product) -> str:
    serialized = json.dumps(product.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def fact_catalog(product: Product) -> dict:
    facts = {'name': {'text': product.name, 'path': '/name'}}
    if product.manufacturer:
        facts['manufacturer'] = {'text': product.manufacturer, 'path': '/manufacturer'}
    if product.description.strip():
        facts['description'] = {'text': product.description, 'path': '/description'}
    for index, (label, value) in enumerate(product.specs.items()):
        if value is not None:
            facts[f'spec_{index}'] = {'text': f'{label}: {value}',
                                     'path': f'/specs/{pointer_key(label)}'}
    for fact in facts.values():
        fact['char_count'] = len(fact['text'])
    return facts


def model_payload(product: Product) -> dict:
    # Conflicting candidate values, URLs, source documents and credentials are NOT sent.
    return {'facts': fact_catalog(product),
            'unknown_fields': [field_label(issue.field) for issue in collect_issues(product)]}


def expand_template(template: str, facts: dict) -> tuple[str, list[str], set[str]]:
    errors = []
    used = set()
    remainder = REFERENCE.sub('', template)
    if re.search(r'\d', remainder):
        errors.append('Технические числа должны поступать только через ссылки на факты.')
    if '{{' in remainder or '}}' in remainder:
        errors.append('Некорректный формат ссылки на факт.')

    def replacement(match):
        key = match.group(1)
        if key not in facts:
            errors.append(f'Неизвестная ссылка на факт: {key}.')
            return match.group(0)
        used.add(key)
        return facts[key]['text']

    return REFERENCE.sub(replacement, template).strip(), errors, used


def validate_draft(draft: Draft, product: Product) -> tuple[dict, list[str], list[dict]]:
    if draft.insufficient_data:
        errors = [] if draft.reason.strip() else ['Не указана причина недостаточности данных.']
        return {}, errors, []
    facts = fact_catalog(product)
    fields = {'seo_title': draft.seo_title, 'seo_description': draft.seo_description,
              'commercial': draft.commercial}
    expanded, errors, evidence = {}, [], []
    for field, template in list(fields.items()) + [
            (f'search_queries.{index}', query) for index, query in enumerate(draft.search_queries)]:
        text, problems, used = expand_template(template, facts)
        expanded[field] = text
        errors.extend(f'{field}: {problem}' for problem in problems)
        evidence.extend({'output_field': field, 'input_field': facts[key]['path'],
                         'inserted_text': facts[key]['text']} for key in sorted(used))
    if not 1 <= len(expanded['seo_title']) <= 70:
        errors.append(f'Заголовок: {len(expanded["seo_title"])} символов; нужно от 1 до 70.')
    if not DESCRIPTION_MIN <= len(expanded['seo_description']) <= DESCRIPTION_MAX:
        errors.append(f'Описание: {len(expanded["seo_description"])} символов; нужно от {DESCRIPTION_MIN} до {DESCRIPTION_MAX}.')
    if not 20 <= len(expanded['commercial']) <= 700:
        errors.append('Commercial-блок должен содержать от 20 до 700 символов.')
    queries = [expanded[f'search_queries.{i}'] for i in range(len(draft.search_queries))]
    if not 3 <= len(queries) <= 5 or any(not query for query in queries):
        errors.append('Нужно от 3 до 5 непустых поисковых запросов.')
    if len({query.casefold() for query in queries}) != len(queries):
        errors.append('Поисковые запросы должны различаться.')
    if draft.reason.strip():
        errors.append('У готового черновика reason должен быть пустым.')
    card = {field: expanded[field] for field in fields}
    card['search_queries'] = queries
    # Technical table is never taken from the generated answer.
    card['specifications'] = specification_rows(product)
    return card, list(dict.fromkeys(errors)), evidence


def _usage_add(totals: dict, response):
    usage = getattr(response, 'usage', None)
    if usage is None:
        totals['complete'] = False
        return
    totals['input_tokens'] += usage.input_tokens
    totals['output_tokens'] += usage.output_tokens
    totals['cached_input_tokens'] += getattr(getattr(usage, 'input_tokens_details', None), 'cached_tokens', 0) or 0
    totals['reasoning_tokens'] += getattr(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens', 0) or 0


def _estimate_cost(usage):
    if not usage['complete']:
        return None
    return round(((usage['input_tokens'] - usage['cached_input_tokens']) * 0.10
                  + usage['cached_input_tokens'] * 0.01 + usage['output_tokens'] * 0.50) / 1_000_000, 8)


def _request(client, messages):
    try:
        return client.responses.create(
            model=MODEL, instructions=SYSTEM_PROMPT, input=messages,
            reasoning={'effort': REASONING_EFFORT}, max_output_tokens=MAX_OUTPUT_TOKENS,
            service_tier='default', store=False,
            text={'format': {'type': 'json_schema', 'name': 'product_draft',
                             'strict': True, 'schema': Draft.model_json_schema()}},
        )
    except AuthenticationError:
        raise GenerationError('OpenAI отклонил ключ. Проверьте OPENAI_API_KEY в .env.') from None
    except PermissionDeniedError:
        raise GenerationError('У ключа нет доступа к GPT-6 Luna или запрос запрещён для этого проекта.') from None
    except RateLimitError:
        raise GenerationError('OpenAI ограничил запросы. Проверьте баланс, квоту и лимиты проекта; затем повторите.') from None
    except APITimeoutError:
        raise GenerationError('OpenAI не ответил вовремя. Запрос мог быть обработан; автоматического повтора нет.') from None
    except APIConnectionError:
        raise GenerationError('Не удалось подключиться к OpenAI. Проверьте доступ к api.openai.com.') from None
    except APIStatusError as error:
        raise GenerationError(f'OpenAI вернул HTTP {error.status_code}. Повторите позже или проверьте настройки API.') from None


def generate_card(product: Product, *, client=None) -> dict:
    """At most two content attempts. No automatic model switch or SDK retries."""
    result = {
        'model': MODEL, 'reasoning_effort': REASONING_EFFORT, 'prompt_version': PROMPT_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': fingerprint(product),
        'status': 'needs_review', 'card': None, 'validation_messages': [], 'evidence': [],
        'missing_information': [issue.model_dump(mode='json') for issue in collect_issues(product)],
        'sources': [source.model_dump(mode='json') for source in product.sources],
        'attempts': 0, 'usage': {'input_tokens': 0, 'output_tokens': 0,
                               'cached_input_tokens': 0, 'reasoning_tokens': 0, 'complete': True},
        'estimated_cost_usd': 0.0,
        'review_note': 'Черновик. Автопроверки не доказывают смысловую достоверность: перед публикацией проверьте текст и спорные поля.',
    }
    if not product.description.strip() and not any(value is not None for value in product.specs.values()):
        result['status'] = 'insufficient_data'
        result['validation_messages'] = ['Добавьте описание или технические характеристики. Одного названия недостаточно.']
        return result
    payload = json.dumps(model_payload(product), ensure_ascii=False)
    if len(payload) > MAX_PAYLOAD_CHARS:
        raise GenerationError('Для генерации оставьте до 32 000 символов исходных фактов; сократите фрагмент паспорта.')
    owned_client = client is None
    if owned_client:
        key = api_key()
        if not key:
            raise GenerationError('Добавьте OPENAI_API_KEY в .env, затем повторите генерацию.')
        client = OpenAI(api_key=key, base_url='https://api.openai.com/v1', timeout=60, max_retries=0)
    messages = [{'role': 'user', 'content': payload}]
    try:
        for attempt in range(2):
            response = _request(client, messages)
            result['attempts'] += 1
            _usage_add(result['usage'], response)
            result['estimated_cost_usd'] = _estimate_cost(result['usage'])
            refused = any(getattr(content, 'type', None) == 'refusal'
                          for item in response.output for content in (getattr(item, 'content', None) or []))
            if refused:
                result['validation_messages'] = ['Модель отказалась формировать описание для этих данных.']
                return result
            if response.status != 'completed':
                result['validation_messages'] = ['Ответ модели не завершён. Черновик не сформирован; сократите входные данные и повторите.']
                return result
            raw = response.output_text
            adjustment = None
            try:
                draft = Draft.model_validate_json(raw)
                card, errors, evidence = validate_draft(draft, product)
                # Do not hide invalid facts or other errors by deleting text.
                if (len(errors) == 1 and errors[0].startswith('Описание:')
                        and len(card['seo_description']) > DESCRIPTION_MAX):
                    facts = fact_catalog(product)
                    shorter, adjustment = shorten_description(
                        draft.seo_description, lambda value: expand_template(value, facts))
                    if adjustment:
                        draft = draft.model_copy(update={'seo_description': shorter})
                        card, errors, evidence = validate_draft(draft, product)
            except (ValidationError, ValueError):
                errors = ['Ответ не соответствует JSON-схеме карточки. Верни все поля с корректными типами.']
                draft, card, evidence = None, None, []
            result['validation_messages'] = errors
            if not errors:
                if draft.insufficient_data:
                    result['status'] = 'insufficient_data'
                    result['validation_messages'] = [draft.reason]
                else:
                    result.update(status='ready', card=card, evidence=evidence)
                    if adjustment:
                        result['description_adjustment'] = adjustment
                return result
            if attempt == 0:
                # Bounded repair uses the same trusted instructions and input facts.
                feedback = {'errors': errors, 'target_description_characters': 1000}
                if card:
                    feedback['expanded_description'] = card['seo_description']
                    feedback['actual_description_characters'] = len(card['seo_description'])
                    feedback['instruction'] = (
                        'Перепиши описание примерно на 1000 символов после подстановки. '
                        'Оставь только 2–4 коротких технических факта; полная таблица уже есть отдельно. '
                        'Учитывай char_count каждого факта. Не дублируй подписи характеристик. '
                        'В новом ответе по-прежнему используй ссылки, а не цифры из развёрнутого текста.')
                messages.extend([{'role': 'assistant', 'content': raw[:20000]},
                                 {'role': 'user', 'content': 'Исправь ответ. Замечания проверки: '
                                  + json.dumps(feedback, ensure_ascii=False)}])
        return result
    finally:
        if owned_client:
            client.close()


def result_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + '\n'


def result_markdown(result: dict) -> str:
    text = escape_markdown
    card = result['card']
    if not card:
        return '# Карточка не сформирована\n\n' + '\n'.join(text(m) for m in result['validation_messages']) + '\n'
    lines = [f'# {text(card["seo_title"])}', '', '*Черновик для проверки контент-менеджером.*', '',
             '## SEO-описание', '', text(card['seo_description']), '', '## Характеристики', '',
             '| Характеристика | Значение |', '| --- | --- |']
    lines.extend(f'| {text(row["Характеристика"])} | {text(row["Значение"])} |' for row in card['specifications'])
    lines.extend(['', '## Поисковые запросы', ''])
    lines.extend(f'- {text(query)}' for query in card['search_queries'])
    lines.extend(['', '## Для менеджера', '', text(card['commercial'])])
    if result['missing_information']:
        lines.extend(['', '## Требует уточнения', ''])
        lines.extend(f'- {text(field_label(issue["field"]))}: {text(issue["message"])} '
                     f'{text(issue["question"])}' for issue in result['missing_information'])
    lines.extend(['', '## Источники', ''])
    lines.extend(f'- {text(source["id"])}: {source["url"]}' for source in result['sources'])
    lines.extend(['', text(result['review_note'])])
    if result.get('description_adjustment'):
        adjustment = result['description_adjustment']
        lines.extend(['', f'Описание сокращено по границе предложения: '
                      f'{adjustment["original_characters"]} → {adjustment["final_characters"]} символов.'])
    return '\n'.join(lines) + '\n'
