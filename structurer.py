import json
import re
import time

CHARS_PER_CHUNK = 5000  # ~2000 tokens of text — fits under both Groq and Gemini free limits

QUESTIONS_PROMPT = """You are extracting problems and questions from a Control Systems textbook PDF.

From the raw extracted text, identify each individual problem or question.

Return a JSON array where each object has exactly these fields:
- "question_number": string (e.g. "P1.1", "E2.3", "1")
- "question": string (complete question text, preserve all sub-parts)
- "type": one of "problem", "theory", "design", "mcq"
- "difficulty": one of "easy", "medium", "hard"
- "topics": array of strings (e.g. ["transfer function", "stability", "PID"])
- "has_diagram": boolean (true if the question references a figure or diagram)
- "figure_reference": string or null (e.g. "Figure P1.1", "Fig. 3.2")
- "page": number (page where question starts)
- "answer": null
- "images": []

Return ONLY a JSON array wrapped in ```json fences. No explanation, no other text."""

SOLUTIONS_PROMPT = """You are extracting solutions from a Control Systems solution manual PDF.

From the raw extracted text, identify each solution entry.

Return a JSON array where each object has exactly these fields:
- "question_number": string matching the question number (e.g. "P1.1", "E2.3")
- "answer": string (complete solution/answer text)
- "page": number

Return ONLY a JSON array wrapped in ```json fences. No explanation, no other text."""


def chunk_text(text: str, max_chars: int = CHARS_PER_CHUNK) -> list[str]:
    pages = re.split(r'(?=\n\n--- PAGE \d+ ---\n)', text)
    pages = [p for p in pages if p.strip()]

    chunks, current = [], ""
    for page in pages:
        if len(current) + len(page) > max_chars and current:
            chunks.append(current.strip())
            current = page
        else:
            current += page

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text[:max_chars]]


def _parse_json_response(content: str) -> list:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    json_str = fenced.group(1).strip() if fenced else content.strip()

    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        arr_match = re.search(r"\[[\s\S]*\]", json_str)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except Exception:
                pass
    return []


def call_groq(client, prompt: str, text: str, max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Extract from this text:\n\n{text}"},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            return _parse_json_response(response.choices[0].message.content or "")
        except Exception as e:
            if ("429" in str(e) or "rate_limit" in str(e)) and attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                raise
    return []


def call_gemini(model, prompt: str, text: str, max_retries: int = 3) -> list:
    import google.generativeai as genai
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                f"{prompt}\n\nExtract from this text:\n\n{text}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                ),
            )
            return _parse_json_response(response.text or "")
        except Exception as e:
            if ("429" in str(e) or "quota" in str(e).lower()) and attempt < max_retries - 1:
                time.sleep(30 * (attempt + 1))
            else:
                raise
    return []


def match_solutions_to_questions(questions: list, solutions: list) -> list:
    sol_map = {s["question_number"]: s["answer"] for s in solutions}
    for q in questions:
        q["answer"] = sol_map.get(q.get("question_number", ""))
    return questions
