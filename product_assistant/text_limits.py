"""Conservative shortening of a templated description, before fact expansion."""
import re

DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200

# Do not remove a tail that may qualify an earlier statement. This is a
# conservative text heuristic, not a semantic proof; editorial review remains.
QUALIFIER = re.compile(
    r'\b(?:не|нет|без|только|лишь|при|если|кроме|однако|но|иначе|нельзя|'
    r'огранич\w*|услов\w*|погреш\w*|точност\w*|завис\w*|допуст\w*|'
    r'исключ\w*|предупреж\w*|кратковремен\w*|краткосроч\w*|'
    r'not|no|only|unless|except|however|without|if|accuracy|limit\w*)\b',
    re.IGNORECASE,
)
BOUNDARY = re.compile(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z«"{])')
ABBREVIATION = re.compile(r'\b(?:т|д|п|др|пр|им|ул|г|гг|см|напр|отн|влажн)\.$', re.IGNORECASE)


def shorten_description(template, expand):
    """Keep the longest whole-sentence prefix within bounds, or change nothing.

    `expand` resolves references to source facts. Never split a reference or
    remove a referenced fact, a suspected restriction, or an incomplete ending.
    Only a description whose sole validation error is excess length may call
    this helper. The caller must validate the entire revised draft again.
    """
    original, problems, _ = expand(template)
    if problems or len(original) <= DESCRIPTION_MAX:
        return template, None
    for boundary in reversed(list(BOUNDARY.finditer(template))):
        prefix = template[:boundary.start()].strip()
        tail = template[boundary.end():]
        # References stay opaque, so dots/abbreviations inside their expanded
        # values cannot become sentence boundaries.
        if '{{' in tail or '}}' in tail or QUALIFIER.search(tail):
            continue
        if ABBREVIATION.search(prefix):
            continue
        if any(prefix.count(left) != prefix.count(right)
               for left, right in [('(', ')'), ('[', ']'), ('«', '»')]):
            continue
        if prefix.count('"') % 2:
            continue
        text, errors, _ = expand(prefix)
        if not errors and DESCRIPTION_MIN <= len(text) <= DESCRIPTION_MAX:
            return prefix, {
                'method': 'whole_sentence_tail',
                'original_characters': len(original),
                'final_characters': len(text),
            }
    return template, None
