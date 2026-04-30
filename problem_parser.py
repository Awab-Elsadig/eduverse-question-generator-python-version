import re

# Matches P1.1, E2.3, AP3.4, DP5.1, CP2.1, EP1.2
PROBLEM_RE = re.compile(r'\b((?:AP|DP|CP|EP|P|E)\d+\.\d+)\b')

FIGURE_RE = re.compile(
    r'\b(figure|fig\.?|diagram|block\s+diagram|circuit|sketch|graph|plot|table)\b',
    re.IGNORECASE,
)

PAGE_MARKER_RE = re.compile(r'--- PAGE (\d+) ---')


def parse_problems(text: str) -> list[dict]:
    page_positions = [
        (m.start(), int(m.group(1)))
        for m in PAGE_MARKER_RE.finditer(text)
    ]

    def page_at(pos: int) -> int:
        result = 1
        for char_pos, page_num in page_positions:
            if char_pos <= pos:
                result = page_num
            else:
                break
        return result

    matches = list(PROBLEM_RE.finditer(text))
    problems = []

    for i, m in enumerate(matches):
        q_num = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        page = page_at(start)

        fig_hits = FIGURE_RE.findall(body)
        has_diagram = bool(fig_hits)
        figure_keywords = sorted({kw.lower().rstrip('.') for kw in fig_hits})

        problems.append({
            'question_number': q_num,
            'question': body,
            'page': page,
            'has_diagram': has_diagram,
            'figure_keywords': figure_keywords,
            'answer': None,
            'images': [],
        })

    return problems
