"""Prepare product data and generate reviewed drafts through GPT-6 Luna."""

import json
from pathlib import Path

import streamlit as st

from product_assistant.data import (
    MAX_INPUT_BYTES, InputError, collect_issues, export_json, export_markdown,
    field_label, parse_product, specification_rows,
)
from product_assistant.generation import (
    MODEL, GenerationError, fingerprint, generate_card, result_json, result_markdown,
)
from product_assistant.settings import api_key

ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    'Testo · поверхностный термометр': ROOT / 'data/products/testo-0560-1109.json',
    'APPA 61 · цифровой мультиметр': ROOT / 'data/products/appa-61.json',
    'Testo 608-H1 · термогигрометр': ROOT / 'data/products/testo-608-h1.json',
    'Testo 608-H2 · термогигрометр с сигнализацией': ROOT / 'data/products/testo-608-h2.json',
    'Свой товар': None,
}
EMPTY_PRODUCT = json.dumps({'name': '', 'manufacturer': None, 'specs': {}, 'description': ''},
                           ensure_ascii=False, indent=2)


def example_text(label):
    path = EXAMPLES[label]
    return path.read_text(encoding='utf-8') if path else EMPTY_PRODUCT


def store_draft():
    st.session_state.drafts[st.session_state.active_product] = st.session_state.editor


def switch_product():
    store_draft()
    label = st.session_state.product_choice
    st.session_state.active_product = label
    st.session_state.editor = st.session_state.drafts.get(label, example_text(label))
    st.session_state.upload_error = None


def reset_draft():
    label = st.session_state.active_product
    st.session_state.editor = example_text(label)
    store_draft()


def import_file():
    upload = st.session_state.upload
    if upload is None:
        st.session_state.upload_error = 'Сначала выберите JSON-файл.'
        return
    if upload.size > MAX_INPUT_BYTES:
        st.session_state.upload_error = 'Файл слишком большой. Максимальный размер — 1 МБ.'
        return
    try:
        content = upload.getvalue().decode('utf-8-sig')
        parse_product(content)
    except UnicodeDecodeError:
        st.session_state.upload_error = 'Файл должен быть сохранён в кодировке UTF-8.'
        return
    except InputError as error:
        st.session_state.upload_error = str(error)
        return
    store_draft()
    st.session_state.product_choice = 'Свой товар'
    st.session_state.active_product = 'Свой товар'
    st.session_state.editor = content
    store_draft()
    st.session_state.upload_error = None


st.set_page_config(page_title='Карточка прибора', page_icon='📋', layout='wide')

if 'drafts' not in st.session_state:
    first = next(iter(EXAMPLES))
    st.session_state.drafts = {}
    st.session_state.active_product = first
    st.session_state.editor = example_text(first)
    st.session_state.upload_error = None

if 'generation_results' not in st.session_state:
    st.session_state.generation_results = {}

with st.sidebar:
    st.title('Карточка прибора')
    st.caption('Помощник контент-менеджера')
    st.selectbox('Выберите товар', list(EXAMPLES), key='product_choice', on_change=switch_product)
    st.button('Сбросить правки выбранного товара', on_click=reset_draft, width='stretch')
    st.divider()
    st.file_uploader('Загрузить свой JSON', type=['json'], key='upload',
                     help='UTF-8, до 1 МБ. После выбора нажмите «Открыть файл».')
    st.button('Открыть файл', on_click=import_file, width='stretch')
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    st.caption('Правки сохраняются между переключениями в этой сессии. '
               'Скачайте JSON, чтобы сохранить их после закрытия страницы.')
    st.divider()
    key_available = bool(api_key())
    if key_available:
        st.caption('API-ключ настроен. Запрос отправляется только по кнопке генерации.')
    else:
        st.info('Для генерации добавьте OPENAI_API_KEY в .env.')

st.title('Подготовьте данные для карточки')
st.write('Проверьте характеристики и уточните спорные сведения перед созданием описания.')
left, right = st.columns([1, 1.2], gap='large')

with left:
    st.subheader('Исходные данные')
    st.text_area('Данные товара в JSON', key='editor', height=570, on_change=store_draft,
                 help='Вносите значения вместе с единицами. Неизвестные значения оставляйте null.')
    st.caption('Изменения применяются после выхода из поля или нажатия Ctrl+Enter. '
               'Исходные файлы примеров остаются на диске без изменений.')

product = None
with right:
    try:
        product = parse_product(st.session_state.editor)
    except InputError as error:
        st.error(str(error))
        st.info('Исправьте данные в редакторе. Просмотр и скачивание появятся после проверки.')
    else:
        issues = collect_issues(product)
        st.success('Структура данных корректна')
        st.caption('Значения и ссылки требуют проверки человеком. Приложение проверяет формат данных.')
        values, questions = st.columns(2)
        values.metric('Заполнено характеристик', sum(value is not None for value in product.specs.values()))
        questions.metric('Требует уточнения', len(issues))
        details_tab, issues_tab, sources_tab = st.tabs(['Данные товара', f'Уточнения · {len(issues)}', 'Источники'])
        with details_tab:
            st.subheader(product.name)
            st.text(f'Производитель: {product.manufacturer or "данных нет"}')
            st.text(product.description or 'Описание: данных нет.')
            if product.specs:
                st.dataframe(specification_rows(product), hide_index=True, width='stretch',
                             column_config={'Характеристика': st.column_config.TextColumn(width='medium'),
                                            'Значение': st.column_config.TextColumn(width='large')})
            else:
                st.warning('Добавьте технические характеристики товара.')
        with issues_tab:
            if not issues:
                st.info('Пропуски в заданных полях не обнаружены. Проверьте полноту характеристик для этого прибора.')
            for issue in issues:
                with st.container(border=True):
                    st.text(field_label(issue.field))
                    st.caption('Источники расходятся' if issue.kind == 'source_conflict' else 'Данных нет')
                    st.text(issue.message)
                    if issue.question:
                        st.text(issue.question)
                    if issue.source_ids:
                        st.caption('Источники: ' + ', '.join(issue.source_ids))
        with sources_tab:
            st.caption('Ссылки указаны во входных данных. Их содержимое не загружается и не проверяется приложением.')
            if not product.sources:
                st.info('Источники не указаны. Добавьте ссылку на карточку или паспорт товара.')
            for source in product.sources:
                with st.expander(source.id):
                    st.link_button('Открыть источник', str(source.url))
                    st.text(source.locator)
                    if source.checked_on:
                        st.caption('Дата проверки в исходных данных: ' + source.checked_on)
                    st.caption(f'Ссылок на поля: {len(source.supports)}')

        st.divider()
        st.subheader('Сохранить исходные данные')
        json_col, markdown_col = st.columns(2)
        with json_col:
            st.download_button('Скачать JSON', export_json(product), file_name='product-input.json',
                               mime='application/json', width='stretch', on_click='ignore')
        with markdown_col:
            st.download_button('Скачать Markdown', export_markdown(product), file_name='product-input.md',
                               mime='text/markdown', width='stretch', on_click='ignore')
        with st.expander('Текст для копирования'):
            st.code(export_markdown(product), language='markdown')

if product is not None:
    st.divider()
    st.subheader('Создать описание')
    st.caption('При ошибке формата или длины возможна одна дополнительная попытка.')
    if st.button('Сформировать SEO-карточку', key='generate', disabled=not key_available,
                 type='primary', width='stretch'):
        st.session_state.generation_results.pop(st.session_state.active_product, None)
        try:
            with st.spinner('Подготовка и проверка черновика…'):
                result = generate_card(product)
            st.session_state.generation_results[st.session_state.active_product] = result
        except GenerationError as error:
            st.error(str(error))
    example_path = EXAMPLES.get(st.session_state.active_product)
    saved_path = ROOT / 'examples/generated' / (example_path.stem + '.json') if example_path else None
    if saved_path and saved_path.is_file():
        if st.button('Показать сохранённый пример', key='show_saved', width='stretch',
                     help='Ранее полученная карточка без нового запроса и расходов API.'):
            try:
                saved = json.loads(saved_path.read_text(encoding='utf-8'))
                if saved['input_sha256'] == fingerprint(product):
                    st.session_state.generation_results[st.session_state.active_product] = saved
                else:
                    st.info('Сохранённый пример относится к другим исходным данным.')
            except (OSError, ValueError, KeyError):
                st.error('Не удалось прочитать сохранённый пример.')
    result = st.session_state.generation_results.get(st.session_state.active_product)
    if result and result['input_sha256'] != fingerprint(product):
        st.info('Исходные данные изменились. Сформируйте карточку заново, чтобы учесть правки.')
        result = None
    if result:
        if result['status'] == 'ready':
            card = result['card']
            st.success('Черновик готов. Проверки формата и длины пройдены.')
            st.caption(result['review_note'])
            st.subheader('SEO-заголовок')
            st.text(card['seo_title'])
            st.caption(f'{len(card["seo_title"])} / 70 символов')
            st.subheader('SEO-описание')
            st.text(card['seo_description'])
            st.caption(f'{len(card["seo_description"])} символов · требуется 800–1200')
            if result.get('description_adjustment'):
                adjustment = result['description_adjustment']
                st.caption(f'Описание сокращено по границе предложения: '
                           f'{adjustment["original_characters"]} → {adjustment["final_characters"]} символов.')
            st.subheader('Поисковые запросы')
            for query in card['search_queries']:
                st.text('• ' + query)
            st.caption('Варианты запросов; частотность не проверялась.')
            st.subheader('Для менеджера')
            st.text(card['commercial'])
            if result['missing_information']:
                st.warning(f'Остаётся уточнить полей: {len(result["missing_information"])}. '
                           'Они перечислены во вкладке «Уточнения» и включены в скачиваемую карточку.')
            download_json, download_md = st.columns(2)
            with download_json:
                st.download_button('Карточка · JSON', result_json(result), file_name='product-card.json',
                                   mime='application/json', on_click='ignore', width='stretch')
            with download_md:
                st.download_button('Карточка · Markdown', result_markdown(result), file_name='product-card.md',
                                   mime='text/markdown', on_click='ignore', width='stretch')
            with st.expander('Карточка для копирования'):
                st.code(result_markdown(result), language='markdown')
        else:
            st.warning('Недостаточно данных для описания.' if result['status'] == 'insufficient_data'
                       else 'Черновик не прошёл проверку.')
            for message in result['validation_messages']:
                st.text(message)
