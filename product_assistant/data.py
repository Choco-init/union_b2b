"""Validate input without inventing, coercing or fetching technical facts."""

import json
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MAX_INPUT_BYTES = 1_000_000


class InputError(ValueError):
    """An input error that can be displayed directly in the UI."""


def pointer_key(value: str) -> str:
    return value.replace('~', '~0').replace('/', '~1')


def resolve_pointer(document: dict, pointer: str):
    if not pointer.startswith('/'):
        raise ValueError('Путь поля должен начинаться с /')
    value = document
    for key in pointer[1:].split('/'):
        value = value[key.replace('~1', '/').replace('~0', '~')]
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Source(StrictModel):
    id: str = Field(min_length=1)
    url: AnyHttpUrl
    type: str = ''
    checked_on: str = ''
    locator: str = ''
    local_snapshot: str | None = None
    supports: list[str] = Field(default_factory=list)


class DataIssue(StrictModel):
    field: str
    kind: Literal['missing', 'source_conflict']
    message: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    question: str = ''


class Product(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    id: str | None = None
    name: str
    manufacturer: str | None = None
    identifiers: dict[str, str | None] = Field(default_factory=dict)
    specs: dict[str, str | None] = Field(default_factory=dict)
    description: str = ''
    sources: list[Source] = Field(default_factory=list)
    data_issues: list[DataIssue] = Field(default_factory=list)

    @field_validator('name')
    @classmethod
    def nonempty_name(cls, value):
        if not value.strip():
            raise ValueError('Укажите название товара')
        return value.strip()

    @field_validator('manufacturer', mode='before')
    @classmethod
    def empty_manufacturer(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator('specs', 'identifiers')
    @classmethod
    def validate_mapping(cls, value):
        if any(not key.strip() for key in value):
            raise ValueError('Название характеристики или идентификатора не может быть пустым')
        return {key: (None if isinstance(item, str) and not item.strip() else item)
                for key, item in value.items()}

    @model_validator(mode='after')
    def validate_references(self):
        document = self.model_dump(mode='json')
        source_ids = {source.id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError('Идентификаторы источников должны быть уникальными')
        for source in self.sources:
            for pointer in source.supports:
                try:
                    resolve_pointer(document, pointer)
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f'Источник {source.id} ссылается на отсутствующее поле: {pointer}')
        for issue in self.data_issues:
            if not set(issue.source_ids) <= source_ids:
                raise ValueError(f'Неизвестный источник в замечании: {issue.field}')
            try:
                value = resolve_pointer(document, issue.field)
            except (KeyError, TypeError, ValueError):
                raise ValueError(f'Замечание ссылается на отсутствующее поле: {issue.field}')
            if value not in (None, '', {}):
                raise ValueError(f'Поле {issue.field} заполнено, но замечание о нём осталось. '
                                 'После проверки значения обновите источники и удалите замечание.')
        return self


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f'Поле «{key}» повторяется. Оставьте одно значение.')
        result[key] = value
    return result


def _reject_constant(value):
    raise InputError(f'Значение {value} недопустимо в JSON.')


def parse_product(text: str) -> Product:
    if len(text.encode('utf-8')) > MAX_INPUT_BYTES:
        raise InputError('Файл слишком большой. Максимальный размер — 1 МБ.')
    if not text.strip():
        raise InputError('Вставьте JSON с данными товара или выберите пример.')
    try:
        value = json.loads(text.lstrip('\ufeff'), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise InputError(f'Некорректный JSON: строка {error.lineno}, столбец {error.colno}. '
                         'Проверьте кавычки, запятые и скобки.') from error
    except RecursionError as error:
        raise InputError('Слишком много вложенных объектов в JSON.') from error
    if not isinstance(value, dict):
        raise InputError('Ожидается один объект товара { ... }, а не список или отдельное значение.')
    try:
        return Product.model_validate(value)
    except ValidationError as error:
        messages = []
        translations = {
            'missing': 'обязательное поле отсутствует',
            'string_type': 'ожидается текст; технические значения указывайте вместе с единицами',
            'dict_type': 'ожидается объект с именами полей и значениями',
            'list_type': 'ожидается список',
            'extra_forbidden': 'неизвестное поле — проверьте его название',
            'url_scheme': 'ссылка должна начинаться с http:// или https://',
            'url_parsing': 'некорректная ссылка на источник',
            'literal_error': 'значение не входит в допустимый список',
        }
        for detail in error.errors(include_url=False, include_input=False)[:8]:
            location = '.'.join(str(part) for part in detail['loc']) or 'Товар'
            message = translations.get(detail['type'], detail['msg'].removeprefix('Value error, '))
            messages.append(f'{location}: {message}')
        raise InputError('\n'.join(messages)) from error


def collect_issues(product: Product) -> list[DataIssue]:
    issues = list(product.data_issues)
    known = {issue.field for issue in issues}
    missing = []
    if not product.manufacturer:
        missing.append(('/manufacturer', 'Производитель'))
    if not product.description.strip():
        missing.append(('/description', 'Описание или фрагмент паспорта'))
    if not product.specs:
        missing.append(('/specs', 'Технические характеристики'))
    for category in ('specs', 'identifiers'):
        for key, value in getattr(product, category).items():
            if value is None:
                missing.append((f'/{category}/{pointer_key(key)}', key))
    for pointer, label in missing:
        if pointer not in known:
            issues.append(DataIssue(field=pointer, kind='missing',
                                    message=f'{label}: данных нет.',
                                    question=f'Уточните поле «{label}».'))
            known.add(pointer)
    return issues


def field_label(pointer: str) -> str:
    labels = {'/manufacturer': 'Производитель', '/description': 'Описание',
              '/specs': 'Характеристики', '/identifiers/vendor_code': 'Артикул производителя'}
    return labels.get(pointer, pointer.rsplit('/', 1)[-1].replace('~1', '/').replace('~0', '~'))


def specification_rows(product: Product) -> list[dict[str, str]]:
    conflicts = {issue.field for issue in product.data_issues if issue.kind == 'source_conflict'}
    return [{'Характеристика': key,
             'Значение': value if value is not None else (
                 'Требует уточнения' if f'/specs/{pointer_key(key)}' in conflicts else 'Данных нет')}
            for key, value in product.specs.items()]


def export_json(product: Product) -> str:
    document = product.model_dump(mode='json')
    document['data_issues'] = [issue.model_dump(mode='json') for issue in collect_issues(product)]
    return json.dumps(document, ensure_ascii=False, indent=2) + '\n'


def escape_markdown(value: str) -> str:
    # Keep imported values as text, including table pipes, HTML and links.
    value = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    for character in ('\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '!', '|'):
        value = value.replace(character, '\\' + character)
    return value.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '<br>')


def export_markdown(product: Product) -> str:
    text = escape_markdown
    lines = [f'# {text(product.name)}', '',
             '*Исходные данные товара. SEO-описание ещё не сформировано.*', '',
             f'Производитель: {text(product.manufacturer or "данных нет")}', '',
             text(product.description) if product.description.strip() else 'Описание: данных нет.', '',
             '## Характеристики', '', '| Характеристика | Значение |', '| --- | --- |']
    lines.extend(f'| {text(row["Характеристика"])} | {text(row["Значение"])} |'
                 for row in specification_rows(product))
    if not product.specs:
        lines.append('| Данных нет | Добавьте характеристики |')
    issues = collect_issues(product)
    if issues:
        lines.extend(['', '## Требует уточнения', ''])
        for issue in issues:
            lines.append(f'- **{text(field_label(issue.field))}**: {text(issue.message)} '
                         f'{text(issue.question)}')
    if product.sources:
        lines.extend(['', '## Указанные источники', '',
                      'Ссылки предоставлены во входных данных; приложение не проверяет их содержимое.', ''])
        for source in product.sources:
            lines.extend([f'- {text(source.id)}: {source.url}', f'  {text(source.locator)}'])
    return '\n'.join(lines) + '\n'
