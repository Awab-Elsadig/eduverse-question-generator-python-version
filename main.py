import asyncio
import base64
import html
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from extractor import extract_page_range

app = FastAPI(title="Problem Extractor")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAVED_PDF      = OUTPUT_DIR / "saved_textbook.pdf"
SAVED_PDF_NAME = OUTPUT_DIR / "saved_textbook_name.txt"

BUNDLED_PDF      = (Path(__file__).parent / "../../Reference and Solution Manual/Modern Control Systems-12 Edition.pdf").resolve()
BUNDLED_PDF_NAME = "Modern Control Systems-12 Edition.pdf"

EDUVERSE_API_URL = os.getenv("EDUVERSE_API_URL", "https://eduverse-team-eduverse-backend.hf.space")
EDUVERSE_EMAIL   = os.getenv("EDUVERSE_EMAIL", "")
EDUVERSE_PASSWORD = os.getenv("EDUVERSE_PASSWORD", "")

_eduverse_token: Optional[str] = None


def _active_pdf() -> Optional[Path]:
    """Return the PDF to use: user-uploaded first, then the bundled textbook."""
    if SAVED_PDF.exists():
        return SAVED_PDF
    if BUNDLED_PDF.exists():
        return BUNDLED_PDF
    return None


async def _get_eduverse_token() -> str:
    global _eduverse_token
    if _eduverse_token:
        return _eduverse_token
    if not EDUVERSE_EMAIL or not EDUVERSE_PASSWORD:
        raise HTTPException(status_code=500, detail="EDUVERSE_EMAIL and EDUVERSE_PASSWORD must be set in .env")
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{EDUVERSE_API_URL}/api/auth/login",
            json={"email": EDUVERSE_EMAIL, "password": EDUVERSE_PASSWORD},
            timeout=15,
        )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"EduVerse login failed: {r.text}")
    data = r.json()
    token = data.get("accessToken") or data.get("access_token") or data.get("token")
    if not token:
        raise HTTPException(status_code=502, detail=f"EduVerse login response missing token: {data}")
    _eduverse_token = token
    return token


GEMINI_PROMPT = """\
You are extracting problems from an academic textbook. The text was extracted via OCR from a scanned PDF and may contain artifacts, merged columns, and garbled spacing. Identify every distinct problem or question.

Return a JSON array where each object has exactly these fields:
- "question_number": string — the problem label as it appears (e.g. "P1.1", "Q3", "Exercise 4.2", "Problem 7")
- "question": string — complete question text, cleaned of OCR errors. Remove figure captions and page headers. Fix broken words from column wrapping. Format all math using LaTeX: inline with $...$ (e.g. $x_1$, $\\omega_n$, $K > 0$), display with $$...$$.
- "page": number — page where the problem starts (use the --- PAGE N --- markers)
- "question_type": string — one of exactly: "written", "mcq", "true_false". Use "mcq" only if the question lists labeled answer choices (A/B/C/D or similar). Use "true_false" only if the question is a direct true-or-false statement. Use "written" for everything else (derivations, calculations, short answer, design problems).
- "difficulty": string — one of exactly: "easy", "medium", "hard". Base this on cognitive load and prerequisite knowledge: easy = recall or single-step, medium = multi-step application, hard = synthesis, design, or proof.
- "bloom_level": string — one of exactly: "remembering", "understanding", "applying", "analyzing", "evaluating", "creating". Pick the highest Bloom's taxonomy level the question primarily demands.
- "has_diagram": boolean — true if the question references a Figure, asks to sketch a diagram or block diagram, or involves a circuit
- "figure_reference": string or null — specific figure label when present (e.g. "Figure 13.18", "Fig. P1.2"), otherwise null
- "exam_ready": boolean — true if suitable for a university exam. Mark false for: pure discussion/describe questions, problems entirely dependent on a figure students won't have, or questions too open-ended to grade objectively.
- "exam_notes": string or null — if exam_ready is false, a brief reason (e.g. "discussion only", "requires Figure P1.2", "open-ended design"). If exam_ready is true, use null.
- "expected_answer": string or null — for "written" and "true_false" question types only: a concise model answer (1–4 sentences or key steps). For "mcq" use null (the correct option already encodes the answer).
- "hints": string or null — a short hint (1–2 sentences) that nudges a student toward the solution without giving it away. Provide for all question types. Use null if no meaningful hint can be given.

Return ONLY a valid JSON array wrapped in ```json fences. No explanation.

--- TEXT START ---
"""

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Problem Extractor</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="window.onKatexReady && window.onKatexReady()"></script>
  <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
  <script>
    // Auto-reload when the dev server restarts (polls /health every 15s)
    (function() {
      var dead = false;
      setInterval(function() {
        fetch('/health', { cache: 'no-store' }).then(function() {
          if (dead) location.reload();
        }).catch(function() { dead = true; });
      }, 15000);
    })();
  </script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.5; }
    .container { max-width: 940px; margin: 0 auto; padding: 28px 16px 100px; }
    h1 { font-size: 1.4rem; font-weight: 700; }
    .subtitle { color: #64748b; margin: 3px 0 24px; font-size: 0.85rem; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 22px; margin-bottom: 12px; }
    .card-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 12px; }
    .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; font-family: inherit; transition: background 0.12s; display: inline-flex; align-items: center; gap: 6px; }
    .btn-primary   { background: #2563eb; color: white; }
    .btn-primary:hover   { background: #1d4ed8; }
    .btn-primary:disabled { background: #93c5fd; cursor: not-allowed; }
    .btn-secondary { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .btn-secondary:hover { background: #e2e8f0; }
    .btn-success   { background: #166534; color: white; }
    .btn-success:hover   { background: #14532d; }
    .btn-danger    { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .btn-sm  { padding: 5px 12px; font-size: 0.78rem; }
    .btn-xs  { padding: 3px 9px;  font-size: 0.72rem; }
    .error-box { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem; display: none; }
    .info-box  { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; font-size: 0.82rem; }
    /* Config bar */
    .config-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
    .config-bar input[type=number] { width: 68px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; font-family: inherit; text-align: center; }
    /* Plain number fields: no stepper arrows, no wheel nudging */
    input[type=number] { -moz-appearance: textfield; appearance: textfield; }
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    .config-bar input:focus { outline: none; border-color: #2563eb; }
    .config-sep { color: #94a3b8; font-size: 0.85rem; }
    #extract-status { font-size: 0.78rem; color: #64748b; }
    /* Spinner (inline) */
    .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.4); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    /* PDF picker */
    .pdf-change-link { font-size: 0.7rem; color: #64748b; text-decoration: underline; flex-shrink: 0; }
    /* Workflow grid */
    .workflow-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .workflow-grid .card { flex: 1; min-width: 260px; margin-bottom: 0; }
    .step-label { font-size: 0.78rem; font-weight: 700; color: #64748b; margin-bottom: 10px; display: flex; align-items: center; gap: 7px; }
    .step-num { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; background: #2563eb; color: white; border-radius: 50%; font-size: 0.7rem; font-weight: 700; flex-shrink: 0; }
    textarea.paste-area { width: 100%; min-height: 130px; resize: vertical; font-size: 0.78rem; font-family: monospace; padding: 9px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; color: #1e293b; margin-bottom: 10px; }
    textarea.paste-area:focus { outline: none; border-color: #2563eb; }
    /* Images strip */
    .images-toggle { font-size: 0.75rem; color: #64748b; cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 5px; padding: 4px 0; margin-bottom: 6px; }
    .images-toggle:hover { color: #334155; }
    /* Filter bar */
    .filter-bar { display: flex; gap: 8px; margin-bottom: 14px; align-items: center; flex-wrap: wrap; }
    .filter-btn { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .filter-btn.active { background: #2563eb; color: white; border-color: #2563eb; }
    .hidden { display: none !important; }
    /* Question sections */
    .q-section { margin-bottom: 18px; }
    .q-section-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #475569; margin: 10px 0 8px; }
    /* Locked state (before PDF chosen) */
    .locked { opacity: 0.4; pointer-events: none; user-select: none; }
    /* PDF picker pill — whole thing is clickable */
    .pdf-picker-label { display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px; background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 6px; font-size: 0.78rem; font-weight: 600; color: #475569; cursor: pointer; transition: border-color 0.15s, background 0.15s, color 0.15s; user-select: none; }
    .pdf-picker-label:hover { border-color: #2563eb; color: #2563eb; background: #eff6ff; }
    .pdf-ready-pill  { display: none; align-items: center; gap: 7px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 5px 11px; font-size: 0.78rem; color: #166534; max-width: 340px; cursor: pointer; }
    .pdf-ready-pill:hover  { background: #dcfce7; }
    .pdf-ready-name  { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
    .pdf-pending-pill { display: none; align-items: center; gap: 7px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 5px 11px; font-size: 0.78rem; color: #92400e; max-width: 380px; cursor: pointer; }
    .pdf-pending-pill:hover { background: #fef9c3; }
    .pdf-pending-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px; }
    body.dark .pdf-picker-label { background: #1e293b; border-color: #475569; color: #94a3b8; }
    body.dark .pdf-picker-label:hover { border-color: #2563eb; color: #93c5fd; background: #1e3a5f; }
    /* Problem cards */
    .q-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; transition: border-color 0.15s; cursor: pointer; }
    .q-card.active-card { border-color: #2563eb; box-shadow: 0 0 0 4px #bfdbfe; background: #f8fbff; }
    .q-card.flash { border-color: #4ade80; box-shadow: 0 0 0 3px #f0fdf4; }
    .q-header { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; margin-bottom: 10px; }
    .q-num { font-weight: 700; font-size: 0.92rem; font-family: monospace; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; }
    .badge { padding: 2px 7px; border-radius: 9999px; font-size: 0.66rem; font-weight: 600; }
    .badge-page  { background: #dbeafe; color: #1d4ed8; }
    .badge-warn  { background: #fef3c7; color: #92400e; }
    .badge-ready { background: #dcfce7; color: #166534; }
    .badge-skip  { background: #f1f5f9; color: #64748b; }
    .q-rendered { font-size: 0.875rem; line-height: 1.9; padding: 2px 0; min-height: 36px; overflow-x: auto; }
    .q-rendered .katex-display { margin: 0.5em 0; }
    .q-text { width: 100%; min-height: 80px; resize: vertical; font-size: 0.82rem; line-height: 1.6; padding: 8px 10px; border: 1px solid #e2e8f0; border-radius: 6px; font-family: inherit; color: #1e293b; background: #fafafa; display: none; }
    .q-text:focus { outline: none; border-color: #2563eb; background: white; }
    .q-controls { display: flex; align-items: center; gap: 7px; margin-top: 8px; flex-wrap: wrap; }
    .q-controls-right { display: inline-flex; align-items: center; gap: 6px; margin-left: auto; }
    .exam-toggle { display: inline-flex; align-items: center; gap: 5px; font-size: 0.73rem; color: #64748b; cursor: pointer; user-select: none; }
    .exam-toggle input[type=checkbox] { cursor: pointer; }
    .q-notes { font-size: 0.71rem; color: #94a3b8; margin-top: 5px; font-style: italic; }
    .q-field-wrap { margin-top: 12px; border-top: 2px solid #e2e8f0; padding-top: 10px; }
    body.dark .q-field-wrap { border-top-color: #334155; }
    .q-field-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
    .q-field-label { display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }
    .q-field-ta { width: 100%; resize: vertical; font-size: 0.82rem; line-height: 1.6; padding: 8px 10px; border: 1px solid #e2e8f0; border-radius: 6px; font-family: inherit; color: #1e293b; background: #fafafa; display: none; margin-top: 4px; }
    .q-field-ta:focus { outline: none; border-color: #2563eb; background: white; }
    .q-field-rendered { font-size: 0.875rem; line-height: 1.8; padding: 2px 0; overflow-x: auto; }
    .q-field-rendered:empty::before { content: '—'; color: #94a3b8; font-style: italic; font-size: 0.8rem; }
    body.dark .q-field-ta { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    body.dark .q-field-ta:focus { background: #0f172a; }
    .chips-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .img-chip { display: inline-flex; align-items: center; gap: 5px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 3px 8px 3px 3px; font-size: 0.72rem; color: #475569; }
    .img-chip img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: zoom-in; border: 1px solid #cbd5e1; }
    .paste-target-pill { margin-left: auto; font-size: 0.68rem; color: #1d4ed8; background: #dbeafe; border: 1px solid #93c5fd; border-radius: 9999px; padding: 2px 7px; display: none; }
    .q-card.active-card .paste-target-pill { display: inline-flex; align-items: center; }
    .img-chip-remove { background: none; border: none; cursor: pointer; color: #94a3b8; font-size: 1rem; line-height: 1; padding: 0; }
    .img-chip-remove:hover { color: #dc2626; }
    /* Export bar */
    .export-bar { position: fixed; bottom: 0; left: 0; right: 0; background: white; border-top: 1px solid #e2e8f0; padding: 8px 24px; display: none; justify-content: space-between; align-items: center; z-index: 100; flex-wrap: wrap; gap: 8px; min-height: 52px; }
    .export-bar-left { display: flex; align-items: center; gap: 14px; font-size: 0.82rem; color: #64748b; }
    .paste-hint { font-size: 0.75rem; color: #2563eb; background: #eff6ff; padding: 2px 8px; border-radius: 4px; }
    /* EduVerse save panel — fixed above export bar */
    .ev-panel { position: fixed; bottom: 52px; left: 0; right: 0; background: #f0fdf4; border-top: 1px solid #bbf7d0; padding: 12px 24px; display: none; z-index: 99; box-shadow: 0 -4px 16px rgba(0,0,0,0.08); }
    .ev-panel.open { display: block; }
    body.dark .ev-panel { background: #052e16; border-color: #166534; box-shadow: 0 -4px 16px rgba(0,0,0,0.4); }
    .ev-panel-title { font-size: 0.78rem; font-weight: 700; color: #166534; margin-bottom: 10px; }
    .ev-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .ev-row input[type=number] { width: 100px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.82rem; font-family: inherit; }
    .ev-row input[type=number]:focus { outline: none; border-color: #16a34a; }
    .ev-status { font-size: 0.78rem; margin-top: 8px; color: #64748b; min-height: 18px; }
    .ev-status.ok  { color: #166534; }
    .ev-status.err { color: #dc2626; }
    body.dark .ev-panel { background: #052e16; border-color: #166534; }
    body.dark .ev-panel-title { color: #86efac; }
    body.dark .ev-row input[type=number] { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    /* Toast */
    .toast { position: fixed; bottom: 60px; left: 50%; transform: translateX(-50%); background: #1e293b; color: white; padding: 7px 16px; border-radius: 6px; font-size: 0.78rem; z-index: 200; pointer-events: none; opacity: 0; transition: opacity 0.2s; }
    .toast.show { opacity: 1; }
    .workflow-busy { display: none; margin-top: 8px; align-items: center; gap: 8px; font-size: 0.78rem; color: #475569; }
    .workflow-busy.show { display: inline-flex; }
    .friendly-loader { display: inline-flex; gap: 4px; align-items: flex-end; }
    .friendly-loader span { width: 6px; height: 6px; border-radius: 9999px; background: #2563eb; animation: friendly-bounce 1s infinite ease-in-out; }
    .friendly-loader span:nth-child(2) { animation-delay: 0.12s; }
    .friendly-loader span:nth-child(3) { animation-delay: 0.24s; }
    @keyframes friendly-bounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.5; } 40% { transform: translateY(-4px); opacity: 1; } }
    .img-modal { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.8); display: none; align-items: center; justify-content: center; z-index: 300; padding: 18px; }
    .img-modal.open { display: flex; }
    .img-modal img { max-width: min(1100px, 96vw); max-height: 90vh; border-radius: 8px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.45); background: white; object-fit: contain; }
    .img-modal-close { position: absolute; top: 14px; right: 14px; border: 1px solid #94a3b8; background: #0f172a; color: white; border-radius: 6px; font-size: 0.78rem; padding: 6px 10px; cursor: pointer; }
    /* Dark mode toggle */
    .dm-toggle { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; border: 1px solid #e2e8f0; background: #f1f5f9; color: #475569; cursor: pointer; transition: background 0.15s, border-color 0.15s, color 0.15s; flex-shrink: 0; }
    .dm-toggle:hover { background: #e2e8f0; color: #1e293b; }
    /* Dark mode */
    body.dark { background: #0f172a; color: #e2e8f0; }
    body.dark .card { background: #1e293b; border-color: #334155; }
    body.dark .btn-secondary { background: #1e293b; color: #94a3b8; border-color: #334155; }
    body.dark .btn-secondary:hover { background: #334155; color: #e2e8f0; }
    body.dark .btn-danger { background: #450a0a; color: #f87171; border-color: #7f1d1d; }
    body.dark .subtitle { color: #64748b; }
    body.dark .card-title { color: #64748b; }
    body.dark .config-sep { color: #475569; }
    body.dark #extract-status { color: #64748b; }
    body.dark .config-bar input[type=number] { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    body.dark .pdf-change-link { color: #64748b; }
    body.dark .info-box { background: #1e3a5f; border-color: #1d4ed8; color: #93c5fd; }
    body.dark .error-box { background: #450a0a; border-color: #7f1d1d; color: #f87171; }
    body.dark .step-label { color: #64748b; }
    body.dark textarea.paste-area { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    body.dark .images-toggle { color: #64748b; }
    body.dark .images-toggle:hover { color: #94a3b8; }
    body.dark .filter-btn { background: #1e293b; color: #64748b; border-color: #334155; }
    body.dark .filter-btn.active { background: #2563eb; color: white; border-color: #2563eb; }
    body.dark .q-section-title { color: #64748b; }
    body.dark .q-card { background: #1e293b; border-color: #334155; }
    body.dark .q-card.active-card { border-color: #2563eb; box-shadow: 0 0 0 4px #1e3a5f; background: #172554; }
    body.dark .q-num { background: #0f172a; color: #93c5fd; }
    body.dark .badge-page { background: #1e3a5f; color: #93c5fd; }
    body.dark .badge-warn { background: #422006; color: #fcd34d; }
    body.dark .badge-ready { background: #14532d; color: #86efac; }
    body.dark .badge-skip { background: #1e293b; color: #64748b; }
    body.dark .q-rendered { color: #e2e8f0; }
    body.dark .q-text { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    body.dark .q-text:focus { background: #0f172a; }
    body.dark .q-answer-body { background: #0f172a; border-color: #334155; }
    body.dark .q-notes { color: #475569; }
    body.dark .exam-toggle { color: #64748b; }
    body.dark .img-chip { background: #0f172a; border-color: #334155; color: #94a3b8; }
    body.dark .img-chip img { border-color: #334155; }
    body.dark .paste-target-pill { background: #1e3a5f; border-color: #1d4ed8; color: #93c5fd; }
    body.dark .export-bar { background: #1e293b; border-color: #334155; }
    body.dark .export-bar-left { color: #64748b; }
    body.dark .export-bar input[type=number], body.dark .export-bar input[type=text] { background: #0f172a; border-color: #334155; color: #e2e8f0; }
    body.dark .paste-hint { background: #1e3a5f; color: #93c5fd; }
    body.dark .dm-toggle { background: #1e293b; border-color: #334155; color: #94a3b8; }
    body.dark .dm-toggle:hover { background: #334155; color: #e2e8f0; }
    body.dark .workflow-busy { color: #64748b; }
  </style>
</head>
<body>
<div class="container">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:3px;">
    <h1>Problem Extractor</h1>
    <div style="display:flex;gap:8px;align-items:center;">
    <a href="/latest-questions?courseId=30&chapterId=15&limit=100" target="_blank"
       class="btn btn-secondary" style="text-decoration:none;font-size:0.78rem;padding:6px 12px;"
       title="Inspect latest saved questions in EduVerse">
      <i data-lucide="database" style="width:13px;height:13px;"></i>
      Latest saved
    </a>
    <button class="dm-toggle" id="dm-toggle-btn" onclick="toggleDark()" title="Toggle dark mode" aria-label="Toggle dark mode">
      <svg id="dm-icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <svg id="dm-icon-sun"  width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    </button>
    </div>
  </div>
  <p class="subtitle">Extract text &rarr; Gemini structures problems &rarr; review &amp; export.</p>

  <div id="error-box" class="error-box"></div>

  <input type="file" id="pdf-file-input" accept=".pdf" style="display:none;" onchange="handlePdfSelect(this)">

  <!-- ── Config bar ── -->
  <div id="config-card" class="card" style="margin-bottom:16px;">
    <div class="config-bar">
      <!-- PDF picker: whole element is clickable -->
      <label id="pdf-picker-label" class="pdf-picker-label" for="pdf-file-input">
        <i data-lucide="book-open" style="width:13px;height:13px;"></i>
        Choose PDF
      </label>
      <label id="pdf-ready-pill" class="pdf-ready-pill" for="pdf-file-input">
        <i data-lucide="check-circle" style="width:13px;height:13px;flex-shrink:0;"></i>
        <span id="pdf-ready-name" class="pdf-ready-name"></span>
        <span class="pdf-change-link">Change</span>
      </label>
      <label id="pdf-pending-pill" class="pdf-pending-pill" for="pdf-file-input">
        <i data-lucide="clock" style="width:13px;height:13px;flex-shrink:0;"></i>
        <span id="pdf-pending-name" class="pdf-pending-name"></span>
        <span style="color:#a16207;font-size:0.68rem;flex-shrink:0;">uploads on Extract</span>
        <span class="pdf-change-link">Change</span>
      </label>
      <!-- Extract controls — locked until PDF chosen -->
      <span id="extract-controls" class="locked" style="display:inline-flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span class="config-sep">|</span>
        <span style="font-size:0.78rem;color:#64748b;">Pages</span>
        <input id="tb-start" type="number" min="1" placeholder="Start">
        <span class="config-sep">&ndash;</span>
        <input id="tb-end" type="number" min="1" placeholder="End">
        <button id="extract-btn" class="btn btn-primary btn-sm" onclick="startExtract()">
          <i data-lucide="scan-text" style="width:13px;height:13px;"></i>
          Extract
        </button>
        <span id="extract-status"></span>
      </span>
    </div>
  </div>

  <!-- ── Workflow — locked until PDF chosen ── -->
  <div id="workflow-section" class="locked">
    <div class="workflow-grid">
      <div class="card">
        <div class="step-label"><span class="step-num">1</span> Copy &amp; paste into <a href="https://gemini.google.com" target="_blank" style="color:#2563eb;">gemini.google.com</a></div>
        <div class="info-box">Copies the full prompt + your extracted text in one click.</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button id="copy-btn" class="btn btn-primary btn-sm" onclick="copyForGemini()"><i data-lucide="clipboard-copy" style="width:13px;height:13px;"></i> Copy prompt + text</button>
          <button id="download-raw-btn" class="btn btn-secondary btn-sm" onclick="downloadRawText()"><i data-lucide="download" style="width:13px;height:13px;"></i> Download .txt</button>
        </div>
      </div>
      <div class="card">
        <div class="step-label"><span class="step-num">2</span> Paste Gemini&rsquo;s JSON response</div>
        <textarea id="json-paste" class="paste-area" placeholder="Paste here (with or without the ```json fences)..."></textarea>
        <button id="load-json-btn" class="btn btn-success btn-sm" onclick="loadFromJSON()"><i data-lucide="list-checks" style="width:13px;height:13px;"></i> Load Problems</button>
      </div>
    </div>
    <div id="workflow-busy" class="workflow-busy">
      <span class="friendly-loader"><span></span><span></span><span></span></span>
      <span>Extracting pages and gathering text...</span>
    </div>
    <!-- Page images strip -->
    <div id="images-toggle" class="images-toggle" onclick="toggleImages()" style="display:none;">
      <span id="images-arrow">&#9654;</span>
      <span id="images-toggle-label">Page images (0)</span>
    </div>
    <div id="images-section" style="display:none;margin-bottom:12px;">
      <div id="img-gallery" style="display:flex;overflow-x:auto;gap:10px;padding:8px 0;"></div>
    </div>
  </div>

  <!-- ── Review (appears after loading JSON) ── -->
  <div id="review-section" style="display:none;">
    <div id="refresh-warning" style="display:flex;align-items:center;gap:10px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:0.8rem;color:#92400e;">
      <i data-lucide="triangle-alert" style="width:15px;height:15px;flex-shrink:0;"></i>
      <span>Your questions live in memory only — <strong>refreshing this page will clear everything.</strong> Export JSON or Save to EduVerse before leaving.</span>
      <button onclick="document.getElementById('refresh-warning').style.display='none'" style="margin-left:auto;background:none;border:none;cursor:pointer;color:#92400e;flex-shrink:0;"><i data-lucide="x" style="width:14px;height:14px;"></i></button>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
      <p id="review-meta" style="font-size:0.82rem;color:#64748b;"></p>
    </div>
    <div class="filter-bar">
      <button class="btn btn-sm filter-btn active" id="filter-all"   onclick="setFilter('all')">All (0)</button>
      <button class="btn btn-sm filter-btn"        id="filter-ready" onclick="setFilter('ready')">Exam ready (0)</button>
    </div>
    <div id="problems-list"></div>
  </div>
</div>

<!-- EduVerse save panel (shown above export bar when open) -->
<div id="ev-panel" class="ev-panel">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span style="font-size:0.78rem;font-weight:700;color:#166534;display:inline-flex;align-items:center;gap:5px;flex-shrink:0;">
      <i data-lucide="database-zap" style="width:13px;height:13px;"></i> Save to EduVerse
    </span>
    <label style="font-size:0.78rem;color:#475569;flex-shrink:0;">Course ID</label>
    <input id="ev-course-id" type="number" min="1" placeholder="e.g. 2" style="width:90px;padding:5px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.82rem;font-family:inherit;">
    <label style="font-size:0.78rem;color:#475569;flex-shrink:0;">Chapter ID</label>
    <input id="ev-chapter-id" type="number" min="1" placeholder="e.g. 1" style="width:90px;padding:5px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.82rem;font-family:inherit;">
    <button class="btn btn-success btn-sm" onclick="saveToEduverse(this)"><i data-lucide="send" style="width:13px;height:13px;"></i> Save exam-ready</button>
    <button class="btn btn-secondary btn-sm" onclick="toggleEvPanel()"><i data-lucide="x" style="width:13px;height:13px;"></i> Cancel</button>
    <span id="ev-status" class="ev-status" style="margin-top:0;"></span>
  </div>
</div>

<!-- Export bar -->
<div id="export-bar" class="export-bar">
  <div class="export-bar-left">
    <span id="q-count"></span>
    <span id="paste-hint" class="paste-hint" style="display:none;">Ctrl+V &rarr; <span id="paste-target-label"></span></span>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <input id="chapter-export" type="number" min="1" placeholder="Chapter" style="width:88px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.78rem;">
    <input id="teammate-export" type="text" placeholder="Teammate (optional)" style="width:160px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.78rem;">
    <button class="btn btn-secondary btn-sm" onclick="exportChapterZip(this)"><i data-lucide="archive" style="width:13px;height:13px;"></i> Export ZIP</button>
    <button class="btn btn-secondary btn-sm" onclick="exportJSON()"><i data-lucide="file-json" style="width:13px;height:13px;"></i> Export JSON</button>
    <button class="btn btn-primary btn-sm"   onclick="toggleEvPanel()"><i data-lucide="database-zap" style="width:13px;height:13px;"></i> Save to EduVerse</button>
  </div>
</div>

<div id="toast" class="toast"></div>
<div id="img-modal" class="img-modal" onclick="closeImageModal(event)">
  <button class="img-modal-close" type="button" onclick="closeImageModal(event)">Close</button>
  <img id="img-modal-preview" src="" alt="Image preview">
</div>

<script>
  var PROMPT = __PROMPT__;
  var katexReady    = false;
  var problems      = [];
  var rawText       = '';
  var activeFilter  = 'all';
  var activeCardIdx = null;
  var pendingPdfFile = null;
  var hasPdf = false;

  // ── PDF gate helpers ──────────────────────────────────────────────────────

  function unlockWorkflow() {
    hasPdf = true;
    document.getElementById('extract-controls').classList.remove('locked');
    document.getElementById('workflow-section').classList.remove('locked');
  }

  // ── PDF state ─────────────────────────────────────────────────────────────

  function initPdfState() {
    // Do NOT auto-activate saved/bundled PDF — require explicit selection each session.
  }

  function handlePdfSelect(input) {
    var file = input.files && input.files[0];
    if (!file) return;
    pendingPdfFile = file;
    document.getElementById('pdf-picker-label').style.display = 'none';
    document.getElementById('pdf-ready-pill').style.display   = 'none';
    document.getElementById('pdf-pending-pill').style.display = 'inline-flex';
    document.getElementById('pdf-pending-name').textContent   = file.name;
    input.value = '';
    unlockWorkflow();
  }

  function showPdfReady(name) {
    pendingPdfFile = null;
    document.getElementById('pdf-picker-label').style.display = 'none';
    document.getElementById('pdf-pending-pill').style.display = 'none';
    document.getElementById('pdf-ready-pill').style.display   = 'inline-flex';
    document.getElementById('pdf-ready-name').textContent     = name;
  }

  // ── Dark mode ─────────────────────────────────────────────────────────────

  function applyDark(dark) {
    document.body.classList.toggle('dark', dark);
    document.getElementById('dm-icon-moon').style.display = dark ? 'none'  : '';
    document.getElementById('dm-icon-sun').style.display  = dark ? ''      : 'none';
  }

  function toggleDark() {
    var isDark = document.body.classList.contains('dark');
    applyDark(!isDark);
    try { localStorage.setItem('dm', !isDark ? '1' : '0'); } catch(e) {}
  }

  function initNumberInputsNoWheel() {
    document.querySelectorAll('input[type="number"]').forEach(function(el) {
      el.addEventListener('wheel', function(e) { e.preventDefault(); }, { passive: false });
    });
  }

  window.addEventListener('beforeunload', function(e) {
    if (problems.length > 0) { e.preventDefault(); e.returnValue = ''; }
  });

  window.addEventListener('DOMContentLoaded', function() {
    initPdfState();
    initNumberInputsNoWheel();
    try {
      var saved = localStorage.getItem('dm');
      var prefersDark = saved !== null ? saved === '1' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyDark(prefersDark);
    } catch(e) {}
    lucide.createIcons();
  });

  var KATEX_OPTS = {
    delimiters: [
      { left: '$$', right: '$$', display: true  },
      { left: '$',  right: '$',  display: false }
    ],
    throwOnError: false
  };

  // ── KaTeX ─────────────────────────────────────────────────────────────────

  window.onKatexReady = function onKatexReady() {
    katexReady = true;
    document.querySelectorAll('.q-rendered, .q-field-rendered').forEach(function(el) {
      renderMathInElement(el, KATEX_OPTS);
    });
  };

  function renderMath(el) {
    if (katexReady && window.renderMathInElement) {
      renderMathInElement(el, KATEX_OPTS);
    }
  }

  // ── Extract ───────────────────────────────────────────────────────────────

  async function startExtract() {
    hideError();
    if (!hasPdf && !pendingPdfFile) { showError('Choose a PDF first.'); return; }
    var tbStartRaw = document.getElementById('tb-start').value.trim();
    var tbEndRaw   = document.getElementById('tb-end').value.trim();
    var tbStart  = parseInt(tbStartRaw, 10);
    var tbEnd    = parseInt(tbEndRaw, 10);
    var isReextract = rawText.trim().length > 0;
    if (!tbStartRaw || !tbEndRaw || Number.isNaN(tbStart) || Number.isNaN(tbEnd)) {
      showError('Please enter both start and end page numbers.');
      return;
    }
    if (tbStart > tbEnd) { showError('Start page must be <= end page.'); return; }

    var btn = document.getElementById('extract-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span> Extracting...';
    document.getElementById('extract-status').textContent = '';
    setWorkflowBusy(true);

    try {
      var form = new FormData();
      form.append('tb_start', tbStart);
      form.append('tb_end',   tbEnd);
      if (pendingPdfFile) { form.append('textbook', pendingPdfFile, pendingPdfFile.name); }
      var res = await fetch('/api/extract', { method: 'POST', body: form });
      if (!res.ok) { var e = await res.json(); throw new Error(e.detail || 'Extraction failed'); }
      var data = await res.json();
      rawText = data.text || '';
      if (pendingPdfFile) { showPdfReady(pendingPdfFile.name); }

      var imgs = data.images || [];
      document.getElementById('extract-status').textContent =
        '✓ ' + (data.char_count || 0).toLocaleString() + ' chars · ' + imgs.length + ' images';
      showToast(
        isReextract
          ? 'Re-extraction complete. Text and images refreshed.'
          : 'Extraction complete.'
      );

      renderImages(imgs);
      document.getElementById('json-paste').value = '';
    } catch(e) {
      showError(e.message);
    } finally {
      setWorkflowBusy(false);
      btn.disabled = false;
      btn.textContent = 'Re-extract';
    }
  }

  function setWorkflowBusy(isBusy) {
    var busy = document.getElementById('workflow-busy');
    var copyBtn = document.getElementById('copy-btn');
    var downloadBtn = document.getElementById('download-raw-btn');
    var loadBtn = document.getElementById('load-json-btn');
    var pasteArea = document.getElementById('json-paste');
    if (busy) busy.classList.toggle('show', Boolean(isBusy));
    if (copyBtn) copyBtn.disabled = Boolean(isBusy);
    if (downloadBtn) downloadBtn.disabled = Boolean(isBusy);
    if (loadBtn) loadBtn.disabled = Boolean(isBusy);
    if (pasteArea) pasteArea.disabled = Boolean(isBusy);
  }

  function renderImages(imgs) {
    var gallery = document.getElementById('img-gallery');
    var toggle  = document.getElementById('images-toggle');
    gallery.innerHTML = '';
    if (!imgs.length) { toggle.style.display = 'none'; return; }
    document.getElementById('images-toggle-label').textContent = 'Page images (' + imgs.length + ')';
    toggle.style.display = 'inline-flex';
    document.getElementById('images-section').style.display = 'none';
    document.getElementById('images-arrow').textContent = '▶';
    imgs.forEach(function(img) {
      var src     = '/output/' + img.filename;
      var wrapper = document.createElement('div');
      wrapper.style.cssText = 'flex-shrink:0;width:100px;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;background:white;';
      var link    = document.createElement('a'); link.href = src; link.target = '_blank';
      var image   = document.createElement('img');
      image.src   = src;
      image.style.cssText = 'width:100px;height:78px;object-fit:contain;background:#f8fafc;display:block;';
      image.onerror = function() { wrapper.style.display = 'none'; };
      var caption = document.createElement('div');
      caption.style.cssText = 'padding:2px 5px;font-size:0.62rem;color:#64748b;';
      caption.textContent   = 'Page ' + img.page;
      link.appendChild(image); wrapper.appendChild(link); wrapper.appendChild(caption);
      link.addEventListener('click', function(e) {
        e.preventDefault();
        openImageModal(src);
      });
      gallery.appendChild(wrapper);
    });
  }

  function toggleImages() {
    var section = document.getElementById('images-section');
    var arrow   = document.getElementById('images-arrow');
    var open    = section.style.display !== 'none';
    section.style.display = open ? 'none' : 'block';
    arrow.textContent     = open ? '▶' : '▼';
  }

  function openImageModal(src) {
    var modal = document.getElementById('img-modal');
    var preview = document.getElementById('img-modal-preview');
    preview.src = src;
    modal.classList.add('open');
  }

  function closeImageModal(e) {
    if (e && e.target && e.target.id !== 'img-modal' && !e.target.classList.contains('img-modal-close')) return;
    var modal = document.getElementById('img-modal');
    var preview = document.getElementById('img-modal-preview');
    modal.classList.remove('open');
    preview.src = '';
  }

  // ── Clipboard ─────────────────────────────────────────────────────────────

  async function copyForGemini() {
    if (!rawText.trim()) {
      showToast('No extracted text yet. Run Extract first, or paste Gemini JSON directly in step 2.');
      return;
    }
    var full = PROMPT + rawText + '\\n--- TEXT END ---';
    var btn  = document.getElementById('copy-btn');
    try { await navigator.clipboard.writeText(full); }
    catch(e) {
      var ta = document.createElement('textarea');
      ta.value = full; ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0;';
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
    }
    btn.textContent = 'Copied!';
    setTimeout(function() { btn.textContent = 'Copy prompt + text'; }, 2000);
  }

  // ── Load JSON ─────────────────────────────────────────────────────────────

  async function loadFromJSON() {
    hideError();
    var raw = document.getElementById('json-paste').value.trim();
    if (!raw) { showError('Paste the JSON response from Gemini first.'); return; }
    try {
      var res = await fetch('/api/parse-json', {
        method: 'POST', body: raw, headers: { 'Content-Type': 'text/plain' },
      });
      if (!res.ok) { var err = await res.json(); throw new Error(err.detail || 'Parse failed'); }
      var parsed = await res.json();
      problems = parsed.map(function(q) {
        var figureReference = q.figure_reference ? String(q.figure_reference) : detectFigureReference(String(q.question || ''));
        return {
          question_number:  String(q.question_number || ''),
          question:         String(q.question || ''),
          page:             q.page || null,
          question_type:    q.question_type || 'written',
          difficulty:       q.difficulty || 'medium',
          bloom_level:      q.bloom_level || 'applying',
          has_diagram:      Boolean(q.has_diagram),
          figure_reference: figureReference || null,
          exam_ready:       q.exam_ready !== false,
          exam_notes:       q.exam_notes || null,
          expected_answer:  q.expected_answer || null,
          hints:            q.hints || null,
          images:           [],
        };
      });
      var ready = problems.filter(function(q) { return q.exam_ready; }).length;
      document.getElementById('review-meta').textContent =
        problems.length + ' problems loaded · ' + ready + ' exam ready';
      updateFilterCounts();
      renderProblems();
      var rs = document.getElementById('review-section');
      rs.style.display = 'block';
      document.getElementById('export-bar').style.display = 'flex';
      lucide.createIcons();
      setTimeout(function() { rs.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 80);
    } catch(e) { showError('Could not parse JSON: ' + e.message); }
  }

  // ── Filter ────────────────────────────────────────────────────────────────

  function setFilter(f) {
    activeFilter = f;
    document.getElementById('filter-all').classList.toggle('active',   f === 'all');
    document.getElementById('filter-ready').classList.toggle('active', f === 'ready');
    applyFilter();
  }

  function applyFilter() {
    document.querySelectorAll('.q-card').forEach(function(card) {
      card.classList.toggle('hidden', activeFilter === 'ready' && card.dataset.examReady !== '1');
    });
    document.querySelectorAll('.q-section').forEach(function(section) {
      var hasVisible = section.querySelector('.q-card:not(.hidden)');
      section.classList.toggle('hidden', !hasVisible);
    });
    ensureActiveCardVisible();
  }

  function updateFilterCounts() {
    var total = problems.length;
    var ready = problems.filter(function(q) { return q.exam_ready; }).length;
    document.getElementById('filter-all').textContent   = 'All (' + total + ')';
    document.getElementById('filter-ready').textContent = 'Exam ready (' + ready + ')';
    document.getElementById('q-count').textContent      = total + ' problems';
  }

  // ── Problems ──────────────────────────────────────────────────────────────

  var QTYPE_LABELS = {
    'mcq':        'Multiple Choice',
    'true_false': 'True / False',
    'written':    'Written / Open-ended',
    'essay':      'Essay',
  };
  var QTYPE_ORDER = ['mcq', 'true_false', 'written', 'essay'];

  function getGroupKey(questionType) {
    var t = String(questionType || '').toLowerCase();
    return QTYPE_LABELS[t] ? t : 'written';
  }

  function getGroupLabel(key) {
    return QTYPE_LABELS[key] || 'Written / Open-ended';
  }

  function detectFigureReference(text) {
    var src = String(text || '');
    var m = src.match(/\\b(?:Figure|Fig\\.?)\\s*([A-Za-z]?\\d+(?:\\.\\d+)*)\\b/i);
    if (!m) return null;
    return 'Figure ' + m[1];
  }

  function renderProblems() {
    var list = document.getElementById('problems-list');
    list.innerHTML = '';
    if (!problems.length) {
      var empty = document.createElement('div');
      empty.className = 'card'; empty.style.cssText = 'text-align:center;color:#64748b;padding:40px;';
      empty.textContent = 'No problems to show.'; list.appendChild(empty); return;
    }

    var grouped = {};
    problems.forEach(function(q, idx) {
      var typeCode = getGroupKey(q.question_type);
      if (!grouped[typeCode]) grouped[typeCode] = [];
      grouped[typeCode].push({ question: q, index: idx });
    });

    var orderedKeys = QTYPE_ORDER.filter(function(k) { return grouped[k] && grouped[k].length; });

    orderedKeys.forEach(function(typeCode) {
      var section = document.createElement('section');
      section.className = 'q-section';
      section.dataset.type = typeCode;

      var title = document.createElement('div');
      title.className = 'q-section-title';
      title.textContent = getGroupLabel(typeCode) + ' (' + grouped[typeCode].length + ')';
      section.appendChild(title);

      grouped[typeCode].forEach(function(item) {
        section.appendChild(buildCard(item.question, item.index));
      });
      list.appendChild(section);
    });

    applyFilter();
  }

  function mkIcon(pathData, size) {
    size = size || 12;
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', size); svg.setAttribute('height', size);
    svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor'); svg.setAttribute('stroke-width', '2.5');
    svg.setAttribute('stroke-linecap', 'round'); svg.setAttribute('stroke-linejoin', 'round');
    svg.style.cssText = 'flex-shrink:0;vertical-align:middle;';
    svg.innerHTML = pathData;
    return svg;
  }
  var IC = {
    check: '<polyline points="20 6 9 17 4 12"/>',
    x:     '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    warn:  '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    edit:  '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
    done:  '<polyline points="20 6 9 17 4 12"/>',
    trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>',
    img:   '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
  };

  function buildCard(q, idx) {
    var card = document.createElement('div');
    card.className = 'q-card';
    card.dataset.idx = String(idx);
    card.dataset.examReady = q.exam_ready ? '1' : '0';
    card.addEventListener('click', function() { setActiveCard(idx, card); });

    // Header
    var header = document.createElement('div'); header.className = 'q-header';
    var numSpan = document.createElement('span');
    numSpan.className = 'q-num'; numSpan.textContent = q.question_number || ('Q' + (idx+1));
    header.appendChild(numSpan);
    if (q.page) {
      var pg = document.createElement('span'); pg.className = 'badge badge-page';
      pg.textContent = 'Page ' + q.page; header.appendChild(pg);
    }
    if (q.question_type && q.question_type !== 'written') {
      var qt = document.createElement('span'); qt.className = 'badge badge-page';
      qt.textContent = q.question_type.replace('_', '/'); header.appendChild(qt);
    }
    if (q.difficulty) {
      var diff = document.createElement('span');
      diff.className = 'badge ' + (q.difficulty === 'hard' ? 'badge-warn' : q.difficulty === 'easy' ? 'badge-ready' : 'badge-skip');
      diff.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';
      diff.title = 'Difficulty';
      diff.appendChild(mkIcon('<path d="M2 20h20M6 20V10l6-8 6 8v10"/>', 11));
      diff.appendChild(document.createTextNode(' ' + q.difficulty));
      header.appendChild(diff);
    }
    if (q.bloom_level) {
      var bl = document.createElement('span'); bl.className = 'badge badge-skip';
      bl.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';
      bl.title = "Bloom's Taxonomy level";
      bl.appendChild(mkIcon('<path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 12V2"/><path d="M12 12l6.5-6.5"/>', 11));
      bl.appendChild(document.createTextNode(' ' + q.bloom_level));
      header.appendChild(bl);
    }
    if (q.figure_reference || q.has_diagram) {
      var dg = document.createElement('span'); dg.className = 'badge badge-warn';
      dg.id = 'diag-badge-' + idx;
      dg.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';
      dg.title = 'References a diagram — attach an image to resolve';
      dg.appendChild(mkIcon(IC.warn, 11));
      dg.appendChild(document.createTextNode(' ' + (q.figure_reference || 'diagram needed')));
      header.appendChild(dg);
    }
    var eb = document.createElement('span');
    eb.id = 'exam-badge-' + idx;
    eb.className = 'badge ' + (q.exam_ready ? 'badge-ready' : 'badge-skip');
    eb.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';
    eb.appendChild(mkIcon(q.exam_ready ? IC.check : IC.x, 11));
    eb.appendChild(document.createTextNode(q.exam_ready ? ' exam ready' : ' skip'));
    header.appendChild(eb);
    var targetPill = document.createElement('span');
    targetPill.className = 'paste-target-pill';
    targetPill.textContent = 'Ctrl+V target';
    header.appendChild(targetPill);
    var sp = document.createElement('div'); sp.style.flex = '1'; header.appendChild(sp);
    var del = document.createElement('button'); del.className = 'btn btn-danger btn-sm'; del.style.cssText = 'display:inline-flex;align-items:center;gap:4px;';
    del.appendChild(mkIcon(IC.trash, 12)); del.appendChild(document.createTextNode(' Delete'));
    del.addEventListener('click', (function(i) {
      return function(e) { e.stopPropagation(); problems.splice(i,1); updateFilterCounts(); renderProblems(); };
    })(idx));
    header.appendChild(del);
    card.appendChild(header);

    // Rendered view
    var renderDiv = document.createElement('div'); renderDiv.className = 'q-rendered';
    renderDiv.textContent = q.question || ''; renderMath(renderDiv); card.appendChild(renderDiv);

    // Edit textarea
    var ta = document.createElement('textarea'); ta.className = 'q-text'; ta.value = q.question || '';
    ta.addEventListener('change', (function(i) { return function(e) { problems[i].question = e.target.value; }; })(idx));
    card.appendChild(ta);

    // Controls
    var controls = document.createElement('div'); controls.className = 'q-controls';
    var ctrlRight = document.createElement('div'); ctrlRight.className = 'q-controls-right';

    var editBtn = document.createElement('button'); editBtn.className = 'btn btn-secondary btn-xs'; editBtn.type = 'button'; editBtn.style.cssText = 'display:inline-flex;align-items:center;gap:4px;';
    editBtn.appendChild(mkIcon(IC.edit, 11)); editBtn.appendChild(document.createTextNode(' Edit'));
    var doneBtn = document.createElement('button'); doneBtn.className = 'btn btn-primary btn-xs'; doneBtn.type = 'button'; doneBtn.style.cssText = 'display:none;align-items:center;gap:4px;';
    doneBtn.appendChild(mkIcon(IC.done, 11)); doneBtn.appendChild(document.createTextNode(' Done'));
    editBtn.addEventListener('click', function(e) {
      e.stopPropagation(); renderDiv.style.display = 'none'; ta.style.display = 'block'; ta.focus();
      editBtn.style.display = 'none'; doneBtn.style.display = 'inline-flex';
    });
    doneBtn.addEventListener('click', (function(i) {
      return function(e) {
        e.stopPropagation(); problems[i].question = ta.value;
        renderDiv.textContent = ta.value; renderMath(renderDiv);
        renderDiv.style.display = 'block'; ta.style.display = 'none';
        editBtn.style.display = 'inline-flex'; doneBtn.style.display = 'none';
      };
    })(idx));

    // Attach diagram
    var fi = document.createElement('input'); fi.type = 'file'; fi.accept = 'image/*'; fi.style.display = 'none';
    fi.addEventListener('change', (function(i) {
      return function(e) {
        var file = e.target.files[0]; if (!file) return;
        var reader = new FileReader();
        reader.onload = function(ev) {
          problems[i].images = problems[i].images || [];
          problems[i].images.push({ data: ev.target.result, name: file.name });
          updateChips(i, chipsRow);
        };
        reader.readAsDataURL(file); e.target.value = '';
      };
    })(idx));
    card.appendChild(fi);
    var attachBtn = document.createElement('button'); attachBtn.className = 'btn btn-secondary btn-xs'; attachBtn.type = 'button'; attachBtn.style.cssText = 'display:inline-flex;align-items:center;gap:4px;';
    attachBtn.appendChild(mkIcon(IC.img, 11)); attachBtn.appendChild(document.createTextNode(' Attach diagram'));
    attachBtn.addEventListener('click', function(e) { e.stopPropagation(); fi.click(); });

    ctrlRight.appendChild(editBtn); ctrlRight.appendChild(doneBtn); ctrlRight.appendChild(attachBtn);
    controls.appendChild(ctrlRight);

    // Exam ready toggle
    var examLabel = document.createElement('label'); examLabel.className = 'exam-toggle';
    var examCb = document.createElement('input'); examCb.type = 'checkbox'; examCb.checked = Boolean(q.exam_ready);
    examCb.addEventListener('change', (function(i, cardEl) {
      return function(e) {
        problems[i].exam_ready = e.target.checked;
        cardEl.dataset.examReady = e.target.checked ? '1' : '0';
        var badge = document.getElementById('exam-badge-' + i);
        if (badge) {
          badge.className = 'badge ' + (e.target.checked ? 'badge-ready' : 'badge-skip');
          badge.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';
          badge.innerHTML = '';
          badge.appendChild(mkIcon(e.target.checked ? IC.check : IC.x, 11));
          badge.appendChild(document.createTextNode(e.target.checked ? ' exam ready' : ' skip'));
        }
        applyFilter(); updateFilterCounts();
      };
    })(idx, card));
    var examLbl = document.createElement('span'); examLbl.textContent = 'Exam ready';
    examLabel.appendChild(examCb); examLabel.appendChild(examLbl);
    ctrlRight.insertBefore(examLabel, editBtn);
    card.appendChild(controls);

    // Notes
    if (q.exam_notes) {
      var notes = document.createElement('div'); notes.className = 'q-notes';
      notes.textContent = 'Note: ' + q.exam_notes; card.appendChild(notes);
    }

    // Helper: build a render+edit field section (same pattern as the question field)
    function makeField(labelIcon, labelText, initialValue, rows, onSave) {
      var wrap = document.createElement('div'); wrap.className = 'q-field-wrap';
      var hdr = document.createElement('div'); hdr.className = 'q-field-header';
      var lbl = document.createElement('div'); lbl.className = 'q-field-label';
      lbl.appendChild(mkIcon(labelIcon, 11)); lbl.appendChild(document.createTextNode(' ' + labelText));
      var fEditBtn = document.createElement('button'); fEditBtn.className = 'btn btn-secondary btn-xs'; fEditBtn.type = 'button';
      fEditBtn.style.cssText = 'display:inline-flex;align-items:center;gap:4px;';
      fEditBtn.appendChild(mkIcon(IC.edit, 11)); fEditBtn.appendChild(document.createTextNode(' Edit'));
      var fDoneBtn = document.createElement('button'); fDoneBtn.className = 'btn btn-primary btn-xs'; fDoneBtn.type = 'button';
      fDoneBtn.style.cssText = 'display:none;align-items:center;gap:4px;';
      fDoneBtn.appendChild(mkIcon(IC.done, 11)); fDoneBtn.appendChild(document.createTextNode(' Done'));
      hdr.appendChild(lbl); hdr.appendChild(fEditBtn); hdr.appendChild(fDoneBtn);
      var rd = document.createElement('div'); rd.className = 'q-field-rendered';
      rd.textContent = initialValue || ''; renderMath(rd);
      var ta = document.createElement('textarea'); ta.className = 'q-field-ta';
      ta.rows = rows; ta.value = initialValue || '';
      fEditBtn.addEventListener('click', function(e) {
        e.stopPropagation(); rd.style.display = 'none'; ta.style.display = 'block'; ta.focus();
        fEditBtn.style.display = 'none'; fDoneBtn.style.display = 'inline-flex';
      });
      fDoneBtn.addEventListener('click', function(e) {
        e.stopPropagation(); onSave(ta.value);
        rd.textContent = ta.value; renderMath(rd);
        rd.style.display = ''; ta.style.display = 'none';
        fEditBtn.style.display = 'inline-flex'; fDoneBtn.style.display = 'none';
      });
      wrap.appendChild(hdr); wrap.appendChild(rd); wrap.appendChild(ta);
      return wrap;
    }

    // Model answer (written / true_false only)
    if (q.question_type === 'written' || q.question_type === 'true_false') {
      card.appendChild(makeField(IC.done, 'Model answer', q.expected_answer, 3, (function(i) {
        return function(v) { problems[i].expected_answer = v; };
      })(idx)));
    }

    // Hint (all question types)
    card.appendChild(makeField(IC.warn, 'Hint', q.hints, 2, (function(i) {
      return function(v) { problems[i].hints = v; };
    })(idx)));

    // Image chips
    var chipsRow = document.createElement('div'); chipsRow.className = 'chips-row';
    chipsRow.id = 'chips-' + idx;
    updateChips(idx, chipsRow); card.appendChild(chipsRow);

    return card;
  }

  function updateChips(idx, chipsRow) {
    chipsRow.innerHTML = '';
    var hasImages = (problems[idx].images || []).length > 0;
    var diagBadge = document.getElementById('diag-badge-' + idx);
    if (diagBadge) {
      if (hasImages) {
        diagBadge.className = 'badge badge-ready';
        diagBadge.title = 'Diagram attached';
        diagBadge.innerHTML = '';
        diagBadge.appendChild(mkIcon(IC.img, 11));
        diagBadge.appendChild(document.createTextNode(' diagram attached'));
      } else {
        diagBadge.className = 'badge badge-warn';
        diagBadge.title = 'References a diagram — attach an image to resolve';
        diagBadge.innerHTML = '';
        diagBadge.appendChild(mkIcon(IC.warn, 11));
        var label = problems[idx].figure_reference || 'diagram needed';
        diagBadge.appendChild(document.createTextNode(' ' + label));
      }
    }
    (problems[idx].images || []).forEach(function(img, ii) {
      var chip = document.createElement('div'); chip.className = 'img-chip';
      var thumb = document.createElement('img'); thumb.src = img.data; thumb.title = img.name;
      thumb.addEventListener('click', function() { openImageModal(img.data); });
      var name = document.createElement('span'); name.textContent = img.name || 'image';
      name.style.cssText = 'max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
      var rm = document.createElement('button'); rm.className = 'img-chip-remove'; rm.textContent = '✕';
      rm.addEventListener('click', (function(i, imgI, row) {
        return function() { problems[i].images.splice(imgI, 1); updateChips(i, row); };
      })(idx, ii, chipsRow));
      chip.appendChild(thumb); chip.appendChild(name); chip.appendChild(rm); chipsRow.appendChild(chip);
    });
  }

  // ── Active card + Ctrl+V paste ────────────────────────────────────────────

  function setActiveCard(idx, cardEl) {
    activeCardIdx = idx;
    document.querySelectorAll('.q-card').forEach(function(c) { c.classList.remove('active-card'); });
    cardEl.classList.add('active-card');
    var hint = document.getElementById('paste-hint');
    var lbl  = document.getElementById('paste-target-label');
    if (hint && lbl) {
      lbl.textContent  = problems[idx] ? problems[idx].question_number : '';
      hint.style.display = 'inline-block';
    }
  }

  function ensureActiveCardVisible() {
    if (activeCardIdx !== null) {
      var current = document.querySelector('.q-card[data-idx="' + activeCardIdx + '"]');
      if (current && !current.classList.contains('hidden')) return;
    }
    var firstVisible = document.querySelector('.q-card:not(.hidden)');
    if (!firstVisible) {
      activeCardIdx = null;
      var hint = document.getElementById('paste-hint');
      if (hint) hint.style.display = 'none';
      return;
    }
    var nextIdx = parseInt(firstVisible.dataset.idx, 10);
    setActiveCard(nextIdx, firstVisible);
  }

  document.addEventListener('paste', function(e) {
    var items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    var imageItem = null;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) { imageItem = items[i]; break; }
    }
    if (!imageItem) return;
    e.preventDefault();
    if (activeCardIdx === null || activeCardIdx >= problems.length) {
      showToast('No target question available for pasted image.');
      return;
    }
    var file = imageItem.getAsFile();
    var reader = new FileReader();
    var capturedIdx = activeCardIdx;
    reader.onload = function(ev) {
      problems[capturedIdx].images = problems[capturedIdx].images || [];
      problems[capturedIdx].images.push({ data: ev.target.result, name: 'pasted.png' });
      var chipsRow = document.getElementById('chips-' + capturedIdx);
      if (chipsRow) updateChips(capturedIdx, chipsRow);
      var card = document.querySelector('.q-card.active-card');
      if (card) { card.classList.add('flash'); setTimeout(function() { card.classList.remove('flash'); }, 700); }
      showToast('Diagram attached to ' + (problems[capturedIdx] ? problems[capturedIdx].question_number : ''));
    };
    reader.readAsDataURL(file);
  });

  // ── Utilities ─────────────────────────────────────────────────────────────

  function showError(msg) {
    var b = document.getElementById('error-box'); b.textContent = msg; b.style.display = 'block';
  }
  function hideError() { document.getElementById('error-box').style.display = 'none'; }

  function showToast(msg) {
    var t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
    setTimeout(function() { t.classList.remove('show'); }, 2500);
  }

  // ── EduVerse save ─────────────────────────────────────────────────────────

  function toggleEvPanel() {
    var panel = document.getElementById('ev-panel');
    panel.classList.toggle('open');
    document.getElementById('ev-status').textContent = '';
    document.getElementById('ev-status').className = 'ev-status';
  }

  async function saveToEduverse(btn) {
    var courseId  = parseInt(document.getElementById('ev-course-id').value, 10);
    var chapterId = parseInt(document.getElementById('ev-chapter-id').value, 10);
    var statusEl  = document.getElementById('ev-status');

    if (!courseId || courseId < 1) { statusEl.textContent = 'Enter a valid Course ID.'; statusEl.className = 'ev-status err'; return; }
    if (!chapterId || chapterId < 1) { statusEl.textContent = 'Enter a valid Chapter ID.'; statusEl.className = 'ev-status err'; return; }

    var toSave = problems.filter(function(q) { return q.exam_ready; });
    if (!toSave.length) { statusEl.textContent = 'No exam-ready questions to save. Toggle "Exam ready" on questions first.'; statusEl.className = 'ev-status err'; return; }

    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span> Saving...';
    var withImages = toSave.filter(function(q) { return (q.images || []).length > 0; }).length;
    statusEl.textContent = 'Sending ' + toSave.length + ' questions to EduVerse' + (withImages ? ' (uploading ' + withImages + ' diagram' + (withImages !== 1 ? 's' : '') + ')' : '') + '...';
    statusEl.className = 'ev-status';

    try {
      var res = await fetch('/api/save-to-eduverse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ courseId: courseId, chapterId: chapterId, problems: toSave }),
      });
      var data = await res.json();
      if (!res.ok) { throw new Error(data.detail || 'Save failed'); }

      var msg = '✓ Saved ' + data.saved + ' question' + (data.saved !== 1 ? 's' : '') + ' to EduVerse as drafts.';
      if (data.imagesUploaded > 0) msg += ' ' + data.imagesUploaded + ' diagram' + (data.imagesUploaded !== 1 ? 's' : '') + ' uploaded.';
      if (data.failed > 0) { msg += ' ' + data.failed + ' failed (see console).'; console.warn('Failed items:', data.failedItems); }
      statusEl.textContent = msg;
      statusEl.className = 'ev-status ok';
      showToast('Saved ' + data.saved + ' questions to EduVerse.');
      document.getElementById('refresh-warning').style.display = 'none';
    } catch(e) {
      statusEl.textContent = 'Error: ' + e.message;
      statusEl.className = 'ev-status err';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Save exam-ready questions';
    }
  }

  function downloadRawText() {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([rawText], { type: 'text/plain' }));
    a.download = 'extracted_text.txt'; a.click();
  }

  function exportJSON() {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(problems, null, 2)], { type: 'application/json' }));
    a.download = 'problems.json'; a.click();
  }

  function inferChapterFromQuestionNumber(questionNumber) {
    var qn = String(questionNumber || '').trim().toUpperCase();
    var m = qn.match(/^[A-Z]+(\\d+)\\./);
    return m ? parseInt(m[1], 10) : null;
  }

  async function exportChapterZip(btn) {
    hideError();
    if (!problems.length) {
      showError('No questions loaded yet.');
      return;
    }
    var chapterRaw = document.getElementById('chapter-export').value.trim();
    var chapter = parseInt(chapterRaw, 10);
    if (!chapterRaw || Number.isNaN(chapter) || chapter < 1) {
      showError('Enter a valid chapter number for export.');
      return;
    }
    var teammate = document.getElementById('teammate-export').value.trim();
    try {
      if (btn) { btn.disabled = true; btn.textContent = 'Packaging...'; }
      var res = await fetch('/api/export-chapter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapter: chapter,
          teammate: teammate || null,
          source_book: 'Modern Control Systems 12e',
          problems: problems,
        }),
      });
      if (!res.ok) {
        var err = await res.json();
        throw new Error(err.detail || 'Export failed');
      }
      var blob = await res.blob();
      var cd = res.headers.get('Content-Disposition') || '';
      var m = cd.match(/filename="([^"]+)"/);
      var filename = m ? m[1] : ('chapter-' + chapter + '-package.zip');
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
      showToast('Chapter ZIP exported.');
    } catch(e) {
      showError(e.message);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = 'Export chapter ZIP'; }
    }
  }
</script>
</body>
</html>"""

HTML = _HTML.replace("__PROMPT__", json.dumps(GEMINI_PROMPT))


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/pdf-info")
def pdf_info():
    if SAVED_PDF.exists():
        name = SAVED_PDF_NAME.read_text(encoding="utf-8") if SAVED_PDF_NAME.exists() else "textbook.pdf"
        return {"saved": True, "name": name}
    if BUNDLED_PDF.exists():
        return {"saved": True, "name": BUNDLED_PDF_NAME}
    return {"saved": False, "name": None}


@app.post("/api/extract")
async def extract(
    textbook: Optional[UploadFile] = File(None),
    tb_start: int = Form(...),
    tb_end: int = Form(...),
):
    if textbook and textbook.filename:
        with open(SAVED_PDF, "wb") as f:
            shutil.copyfileobj(textbook.file, f)
        SAVED_PDF_NAME.write_text(textbook.filename, encoding="utf-8")

    active = _active_pdf()
    if active is None:
        raise HTTPException(status_code=400, detail="No PDF available. Please choose a PDF file first.")

    for old in OUTPUT_DIR.glob("q_page*"):
        old.unlink(missing_ok=True)
    text, images = await asyncio.to_thread(
        extract_page_range, str(active), tb_start, tb_end, OUTPUT_DIR, "q"
    )
    with open(OUTPUT_DIR / "raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return {"text": text, "images": images, "char_count": len(text)}


# LaTeX commands that start with b/f/n/r/t after a single "\". JSON treats \b \f \n \r \t as
# control characters, so "\frac" becomes form feed + "rac" and "\to" becomes tab + "o" unless
# we double the backslash before json.loads. Longer tokens first so e.g. \rightarrow is not split as \right.
_LATEX_JSON_COLLISION_TOKENS = sorted(
    {
        "boldsymbol",
        "textbf",
        "textit",
        "textsf",
        "texttt",
        "textnormal",
        "text",
        "tfrac",
        "theta",
        "Theta",
        "triangle",
        "tilde",
        "times",
        "tanh",
        "tan",
        "tau",
        "to",
        "Rightarrow",
        "Rrightarrow",
        "rightarrow",
        "rightleftharpoons",
        "right",
        "rho",
        "rangle",
        "rfloor",
        "mathrm",
        "mathcal",
        "mathit",
        "mathfrak",
        "mathtt",
        "mathbf",
        "binom",
        "begin",
        "beta",
        "biggl",
        "biggr",
        "bigl",
        "bigr",
        "big",
        "nabla",
        "notin",
        "neq",
        "nu",
        "frac",
        "fbox",
    },
    key=len,
    reverse=True,
)


def _fix_latex_json_escape_collisions(raw: str) -> str:
    out = raw
    for tok in _LATEX_JSON_COLLISION_TOKENS:
        pat = rf"(?<!\\)\\{re.escape(tok)}(?![A-Za-z])"
        # Callable repl: string repl would interpret `\f` etc. inside the replacement.
        out = re.sub(pat, lambda m, t=tok: chr(92) * 2 + t, out)
    return out


@app.post("/api/parse-json")
async def parse_json_route(request: Request):
    raw = (await request.body()).decode("utf-8").strip()
    m = re.search(r"```(?:json)?\n([\s\S]*?)\n?```", raw)
    if m:
        raw = m.group(1).strip()
    raw = _fix_latex_json_escape_collisions(raw)
    # Fix remaining unescaped LaTeX backslashes (\alpha, \le, ...) for strict JSON.
    # (?<![\\]) so we do not treat the second "\" in "\\frac" as starting a new JSON escape.
    fixed = re.sub(
        r'(?<!\\)\\(.)',
        lambda m: m.group(0) if m.group(1) in '"\\/bfnrtu' else '\\\\' + m.group(1),
        raw,
    )
    try:
        data = json.loads(fixed)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array")
    return data


def _chapter_from_question_number(question_number: str) -> Optional[int]:
    m = re.match(r"^[A-Za-z]+(\d+)\.", (question_number or "").strip())
    return int(m.group(1)) if m else None


@app.post("/api/export-chapter")
async def export_chapter(payload: dict):
    chapter = payload.get("chapter")
    problems = payload.get("problems")
    teammate = (payload.get("teammate") or "unknown").strip()
    source_book = (payload.get("source_book") or "Unknown Book").strip()

    if not isinstance(chapter, int) or chapter < 1:
        raise HTTPException(status_code=400, detail="Invalid chapter number.")
    if not isinstance(problems, list) or not problems:
        raise HTTPException(status_code=400, detail="No problems provided for export.")

    chapter_questions = [
        q for q in problems
        if isinstance(q, dict) and _chapter_from_question_number(str(q.get("question_number", ""))) == chapter
    ]
    if not chapter_questions:
        raise HTTPException(status_code=400, detail=f"No questions found for chapter {chapter}.")

    export_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    teammate_slug = re.sub(r"[^a-z0-9]+", "-", teammate.lower()).strip("-") or "unknown"
    package_slug = f"chapter-{chapter:02d}-{teammate_slug}-{export_ts}"

    zip_buffer = io.BytesIO()
    image_counter = 0
    prepared_questions = []
    manifest = {
        "package_version": 1,
        "package_slug": package_slug,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_book": source_book,
        "chapter": chapter,
        "teammate": teammate,
        "question_count": len(chapter_questions),
    }

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for q in chapter_questions:
            q_num = str(q.get("question_number", "")).strip() or f"Q{len(prepared_questions) + 1}"
            q_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", q_num)
            image_paths = []
            for img in q.get("images", []) or []:
                if not isinstance(img, dict):
                    continue
                data_url = str(img.get("data", ""))
                if not data_url.startswith("data:image/") or "," not in data_url:
                    continue
                header, b64 = data_url.split(",", 1)
                ext_match = re.search(r"data:image/([a-zA-Z0-9+.-]+);base64", header)
                ext = (ext_match.group(1) if ext_match else "png").replace("jpeg", "jpg")
                image_counter += 1
                img_path = f"{package_slug}/images/{q_slug}_{image_counter:02d}.{ext}"
                try:
                    zf.writestr(img_path, base64.b64decode(b64))
                    image_paths.append(img_path.replace(f"{package_slug}/", ""))
                except Exception:
                    continue

            prepared_questions.append({
                "source_book": source_book,
                "chapter": chapter,
                "question_number": q_num,
                "question_type": re.match(r"^([A-Za-z]+)", q_num).group(1).upper() if re.match(r"^([A-Za-z]+)", q_num) else None,
                "question_text": str(q.get("question", "")),
                "page": q.get("page"),
                "has_diagram": bool(q.get("has_diagram")),
                "figure_reference": q.get("figure_reference"),
                "exam_ready": q.get("exam_ready") is not False,
                "exam_notes": q.get("exam_notes"),
                "expected_answer": q.get("expected_answer"),
                "assets": image_paths,
                "source": {
                    "teammate": teammate,
                    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            })

        zf.writestr(
            f"{package_slug}/questions.json",
            json.dumps(prepared_questions, indent=2, ensure_ascii=False),
        )
        manifest["asset_count"] = image_counter
        zf.writestr(
            f"{package_slug}/manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    zip_bytes = zip_buffer.getvalue()
    filename = f"{package_slug}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@app.post("/api/save")
async def save(problems: list):
    with open(OUTPUT_DIR / "problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    return {"ok": True}


_BLOOM_MAP = {
    "remembering": "remembering",
    "understanding": "understanding",
    "applying": "applying",
    "analyzing": "analyzing",
    "evaluating": "evaluating",
    "creating": "creating",
}
_DIFFICULTY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}
_TYPE_MAP = {"written": "written", "mcq": "mcq", "true_false": "true_false", "essay": "essay"}


def _to_backend_question(q: dict, course_id: int, chapter_id: int, question_file_id: int | None = None) -> dict:
    question_type = _TYPE_MAP.get(str(q.get("question_type", "")).lower(), "written")
    difficulty    = _DIFFICULTY_MAP.get(str(q.get("difficulty", "")).lower(), "medium")
    bloom_level   = _BLOOM_MAP.get(str(q.get("bloom_level", "")).lower(), "applying")

    payload: dict = {
        "courseId":     course_id,
        "chapterId":    chapter_id,
        "questionType": question_type,
        "difficulty":   difficulty,
        "bloomLevel":   bloom_level,
        "questionText": str(q.get("question", "") or ""),
        "status":       "draft",
    }

    if question_file_id:
        payload["questionFileId"] = question_file_id

    if question_type in ("written", "essay"):
        payload["expectedAnswerText"] = str(q.get("expected_answer", "") or "")

    if q.get("hints"):
        payload["hints"] = str(q["hints"])

    if question_type == "mcq" and q.get("options"):
        payload["options"] = [
            {"optionText": str(o.get("optionText", o) if isinstance(o, dict) else o), "isCorrect": bool(o.get("isCorrect", False)) if isinstance(o, dict) else False}
            for o in q["options"]
        ]
    elif question_type == "true_false":
        payload["options"] = [
            {"optionText": "True",  "isCorrect": True},
            {"optionText": "False", "isCorrect": False},
        ]

    return payload


async def _upload_question_image(client: httpx.AsyncClient, token: str, img_b64: str, filename: str) -> int | None:
    """Upload a base64 image to EduVerse, return fileId or None on failure."""
    try:
        header, data = img_b64.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        img_bytes = base64.b64decode(data)
    except Exception:
        return None

    async def _do_upload(tok: str):
        return await client.post(
            f"{EDUVERSE_API_URL}/api/question-bank/questions/upload-image",
            headers={"Authorization": f"Bearer {tok}"},
            files={"image": (filename, img_bytes, mime)},
        )

    r = await _do_upload(token)
    if r.status_code == 401:
        global _eduverse_token
        _eduverse_token = None
        token = await _get_eduverse_token()
        r = await _do_upload(token)

    if not r.is_success:
        return None

    resp = r.json()
    file_id = resp.get("fileId") or resp.get("data", {}).get("fileId")
    return int(file_id) if file_id else None


@app.post("/api/save-to-eduverse")
async def save_to_eduverse(payload: dict):
    """
    Bulk-save reviewed problems to the EduVerse question bank.
    Expects: { courseId, chapterId, problems: [...] }
    Images (base64 data URLs) in each problem's `images` array are uploaded
    first; the returned fileId is set as questionFileId on the question.
    """
    global _eduverse_token

    course_id  = payload.get("courseId")
    chapter_id = payload.get("chapterId")
    problems   = payload.get("problems", [])

    if not isinstance(course_id, int) or course_id < 1:
        raise HTTPException(status_code=400, detail="courseId must be a positive integer")
    if not isinstance(chapter_id, int) or chapter_id < 1:
        raise HTTPException(status_code=400, detail="chapterId must be a positive integer")
    if not problems:
        raise HTTPException(status_code=400, detail="No problems provided")

    token = await _get_eduverse_token()
    images_uploaded = 0

    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: upload primary image for each problem that has one
        question_file_ids: list[int | None] = []
        for q in problems:
            imgs = q.get("images") or []
            if imgs:
                first = imgs[0]
                file_id = await _upload_question_image(client, token, first.get("data", ""), first.get("name", "diagram.png"))
                question_file_ids.append(file_id)
                if file_id:
                    images_uploaded += 1
            else:
                question_file_ids.append(None)

        # Step 2: batch-create questions with questionFileId already set
        questions = [
            _to_backend_question(q, course_id, chapter_id, question_file_ids[i])
            for i, q in enumerate(problems)
        ]

        BATCH_SIZE = 50
        all_created, all_failed = [], []

        for i in range(0, len(questions), BATCH_SIZE):
            batch = questions[i:i + BATCH_SIZE]
            r = await client.post(
                f"{EDUVERSE_API_URL}/api/question-bank/questions/batch",
                json={"courseId": course_id, "defaultChapterId": chapter_id, "questions": batch},
                headers={"Authorization": f"Bearer {token}"},
            )
            if r.status_code == 401:
                _eduverse_token = None
                token = await _get_eduverse_token()
                r = await client.post(
                    f"{EDUVERSE_API_URL}/api/question-bank/questions/batch",
                    json={"courseId": course_id, "defaultChapterId": chapter_id, "questions": batch},
                    headers={"Authorization": f"Bearer {token}"},
                )
            if not r.is_success:
                raise HTTPException(status_code=502, detail=f"EduVerse batch failed: {r.text}")
            data = r.json()
            all_created.extend(data.get("created", []))
            all_failed.extend(
                [{"rowIndex": f["rowIndex"] + i, **{k: v for k, v in f.items() if k != "rowIndex"}} for f in data.get("failed", [])]
            )

    return {
        "saved":          len(all_created),
        "failed":         len(all_failed),
        "failedItems":    all_failed,
        "imagesUploaded": images_uploaded,
    }


@app.get("/output/{filename}")
def serve_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/latest-questions", response_class=HTMLResponse)
async def latest_questions_shell(courseId: int = 30, chapterId: Optional[int] = None, limit: int = 100):
    """Shell page — loads instantly, then fetches content via AJAX."""
    chapter_param = f"&chapterId={chapterId}" if chapterId is not None else ""
    data_url = f"/latest-questions-data?courseId={courseId}{chapter_param}&limit={limit}"
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Latest Questions — course {courseId}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.5; }}
  body.dark {{ background: #0f172a; color: #e2e8f0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 28px 16px 60px; }}
  h1 {{ font-size: 1.4rem; font-weight: 700; }}
  .btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; font-family: inherit; transition: background 0.12s; display: inline-flex; align-items: center; gap: 6px; }}
  .btn-secondary {{ background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }}
  .btn-secondary:hover {{ background: #e2e8f0; }}
  .btn-sm {{ padding: 5px 12px; font-size: 0.78rem; }}
  .dm-toggle {{ display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; border: 1px solid #e2e8f0; background: #f1f5f9; color: #475569; cursor: pointer; transition: background 0.15s, border-color 0.15s, color 0.15s; flex-shrink: 0; }}
  .dm-toggle:hover {{ background: #e2e8f0; color: #1e293b; }}
  body.dark .btn-secondary {{ background: #1e293b; color: #94a3b8; border-color: #334155; }}
  body.dark .btn-secondary:hover {{ background: #334155; color: #e2e8f0; }}
  body.dark .dm-toggle {{ background: #1e293b; border-color: #334155; color: #94a3b8; }}
  body.dark .dm-toggle:hover {{ background: #334155; color: #e2e8f0; }}

  /* ── Loading scene ── */
  #loading-scene {{
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 80px 20px 100px; gap: 28px;
  }}
  .loader-books {{
    display: flex; gap: 10px; align-items: flex-end; height: 60px;
  }}
  .book {{
    width: 18px; border-radius: 3px 3px 0 0;
    animation: bookBounce 1.1s ease-in-out infinite;
    transform-origin: bottom center;
  }}
  .book:nth-child(1) {{ height: 46px; background: #2563eb; animation-delay: 0s; }}
  .book:nth-child(2) {{ height: 54px; background: #7c3aed; animation-delay: 0.15s; }}
  .book:nth-child(3) {{ height: 38px; background: #059669; animation-delay: 0.30s; }}
  .book:nth-child(4) {{ height: 58px; background: #d97706; animation-delay: 0.45s; }}
  .book:nth-child(5) {{ height: 42px; background: #dc2626; animation-delay: 0.60s; }}
  @keyframes bookBounce {{
    0%, 100% {{ transform: scaleY(1); opacity: 1; }}
    50% {{ transform: scaleY(0.55); opacity: 0.6; }}
  }}
  .loader-dots {{
    display: flex; gap: 7px;
  }}
  .loader-dots span {{
    width: 8px; height: 8px; border-radius: 50%; background: #94a3b8;
    animation: dotPulse 1.4s ease-in-out infinite;
  }}
  .loader-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
  .loader-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
  @keyframes dotPulse {{
    0%, 80%, 100% {{ transform: scale(0.7); opacity: 0.5; }}
    40% {{ transform: scale(1); opacity: 1; background: #2563eb; }}
  }}
  .loader-msg {{
    font-size: 0.9rem; color: #64748b; text-align: center; max-width: 300px; line-height: 1.6;
  }}
  .loader-msg strong {{ color: #1e293b; display: block; font-size: 1rem; margin-bottom: 4px; }}
  body.dark .loader-msg {{ color: #475569; }}
  body.dark .loader-msg strong {{ color: #e2e8f0; }}
  #error-scene {{
    display: none; text-align: center; padding: 60px 20px;
    color: #dc2626; font-size: 0.9rem;
  }}
  body.dark #error-scene {{ color: #f87171; }}
</style>
</head>
<body>
<div class="container">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:20px;">
    <h1>Latest Saved Questions</h1>
    <div style="display:flex;gap:8px;align-items:center;">
      <a href="/" class="btn btn-secondary btn-sm" style="text-decoration:none;">
        <i data-lucide="arrow-left" style="width:13px;height:13px;"></i> Back
      </a>
      <button class="dm-toggle" onclick="toggleDark()" title="Toggle dark mode" aria-label="Toggle dark mode">
        <svg id="dm-icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        <svg id="dm-icon-sun"  width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      </button>
    </div>
  </div>

  <!-- Loading animation -->
  <div id="loading-scene">
    <div class="loader-books">
      <div class="book"></div><div class="book"></div><div class="book"></div>
      <div class="book"></div><div class="book"></div>
    </div>
    <div class="loader-dots">
      <span></span><span></span><span></span>
    </div>
    <div class="loader-msg">
      <strong>Fetching questions...</strong>
      Pulling from the question bank and checking for duplicates.
    </div>
  </div>
  <div id="error-scene"></div>

  <!-- Content injected here -->
  <div id="content"></div>
</div>

<script>
  // Dark mode
  function applyDark(dark) {{
    document.body.classList.toggle('dark', dark);
    document.getElementById('dm-icon-moon').style.display = dark ? 'none' : '';
    document.getElementById('dm-icon-sun').style.display  = dark ? '' : 'none';
  }}
  function toggleDark() {{
    var isDark = document.body.classList.contains('dark');
    applyDark(!isDark);
    try {{ localStorage.setItem('dm', !isDark ? '1' : '0'); }} catch(e) {{}}
  }}
  (function() {{
    try {{
      var saved = localStorage.getItem('dm');
      applyDark(saved !== null ? saved === '1' : window.matchMedia('(prefers-color-scheme: dark)').matches);
    }} catch(e) {{}}
  }})();

  lucide.createIcons();

  // Fetch data
  fetch("{data_url}")
    .then(function(r) {{
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    }})
    .then(function(fragHtml) {{
      document.getElementById('loading-scene').style.display = 'none';
      var el = document.getElementById('content');
      // Use createContextualFragment so <script> tags execute
      var frag = document.createRange().createContextualFragment(fragHtml);
      el.appendChild(frag);
      // Re-run lucide on injected icons
      lucide.createIcons();
      // KaTeX (may already be loaded; retry if deferred scripts aren't done yet)
      function tryRenderMath() {{
        if (window.renderMathInElement) {{
          renderMathInElement(el, {{ delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$',  right: '$',  display: false}}
          ]}});
        }} else {{
          setTimeout(tryRenderMath, 200);
        }}
      }}
      tryRenderMath();
    }})
    .catch(function(err) {{
      document.getElementById('loading-scene').style.display = 'none';
      var es = document.getElementById('error-scene');
      es.style.display = 'block';
      es.innerHTML = '<strong style="font-size:1rem;display:block;margin-bottom:8px;">Failed to load questions</strong>' + err.message
        + '<br><br><a href="" style="color:#2563eb;">Retry</a>';
    }});
</script>
</body></html>"""


@app.get("/latest-questions-data", response_class=HTMLResponse)
async def latest_questions_data(courseId: int, chapterId: Optional[int] = None, limit: int = 100):
    """Data fragment — fetched by the shell page via AJAX."""
    from collections import defaultdict
    from datetime import datetime as _dt, timezone as _tz

    if limit < 1 or limit > 500:
        limit = 100

    PAGE_SIZE = 100
    MAX_PAGES = 20

    questions: list[dict] = []
    total = 0
    token = await _get_eduverse_token()

    async with httpx.AsyncClient() as client:
        for page in range(1, MAX_PAGES + 1):
            params: dict = {"courseId": courseId, "page": page, "limit": PAGE_SIZE}
            if chapterId is not None:
                params["chapterId"] = chapterId
            r = await client.get(
                f"{EDUVERSE_API_URL}/api/question-bank/questions",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if r.status_code == 401:
                global _eduverse_token
                _eduverse_token = None
                token = await _get_eduverse_token()
                r = await client.get(
                    f"{EDUVERSE_API_URL}/api/question-bank/questions",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"List failed page={page}: {r.text}")
            body = r.json()
            if isinstance(body, dict):
                items = body.get("data") or body.get("items") or body.get("questions") or []
                if page == 1:
                    total = int(body.get("total") or len(items))
            elif isinstance(body, list):
                items = body
                if page == 1:
                    total = len(items)
            else:
                items = []
            if not items:
                break
            questions.extend(items)
            if len(questions) >= limit or len(items) < PAGE_SIZE:
                break

    questions.sort(
        key=lambda q: (q.get("createdAt") or "", q.get("id") or 0),
        reverse=True,
    )
    questions = questions[:limit]

    # Duplicate detection
    dup_groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        text = (q.get("questionText") or "").strip().lower()
        if text:
            dup_groups[text].append(q)
    dups = {k: v for k, v in dup_groups.items() if len(v) > 1}
    dup_ids: set[int] = {q.get("id") for items in dups.values() for q in items}

    # Group by save-minute (YYYY-MM-DD HH:MM)
    def minute_key(q):
        raw = q.get("createdAt") or ""
        if not raw:
            return "~no-date"
        try:
            iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            d = _dt.fromisoformat(iso)
            return d.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "~no-date"

    batches: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        batches[minute_key(q)].append(q)
    # Sort batches newest first
    sorted_batches = sorted(batches.items(), key=lambda kv: kv[0], reverse=True)

    def fmt_date(s):
        if not s:
            return "—"
        try:
            iso = s[:-1] + "+00:00" if isinstance(s, str) and s.endswith("Z") else s
            d = _dt.fromisoformat(iso) if isinstance(iso, str) else iso
            return d.strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            return str(s)

    def fmt_minute(key):
        if key == "~no-date":
            return "Unknown time"
        try:
            d = _dt.strptime(key, "%Y-%m-%d %H:%M")
            return d.strftime("%Y-%m-%d %I:%M %p") + " UTC"
        except Exception:
            return key

    def trunc(s, n=240):
        s = s or ""
        return s if len(s) <= n else s[: n - 1] + "…"

    def img_tag(q):
        url = q.get("questionImageUrl")
        if not url:
            return '<span class="muted">—</span>'
        esc = html.escape(url)
        return f'<img src="{esc}" alt="figure" onclick="openModal(\'{esc}\')">'

    def type_badge(v):
        v = str(v or "")
        return f'<span class="badge badge-blue">{html.escape(v)}</span>' if v else '<span class="muted">—</span>'

    def diff_badge(v):
        v = str(v or "")
        c = {"easy": "badge-green", "medium": "badge-blue", "hard": "badge-red"}.get(v.lower(), "badge-gray")
        return f'<span class="badge {c}">{html.escape(v)}</span>' if v else '<span class="muted">—</span>'

    def status_badge(v):
        v = str(v or "")
        c = {"approved": "badge-green", "draft": "badge-gray", "rejected": "badge-red"}.get(v.lower(), "badge-gray")
        return f'<span class="badge {c}">{html.escape(v)}</span>' if v else '<span class="muted">—</span>'

    # Build batch HTML
    batches_html_parts = []
    for key, qs in sorted_batches:
        def _card(q):
            is_dup = q.get("id") in dup_ids
            has_ans = bool((q.get("expectedAnswerText") or "").strip())
            has_hint = bool((q.get("hints") or "").strip())
            card_cls = "q-card dup-card-q" if is_dup else "q-card"
            dup_marker = '<span class="dup-marker-badge"><i data-lucide="copy" style="width:10px;height:10px;"></i> duplicate</span>' if is_dup else ""
            qtext = q.get("questionText") or ""
            ans_text = html.escape(q.get("expectedAnswerText") or "")
            hint_text = html.escape(q.get("hints") or "")
            rid = q.get("id")
            saved_fmt = html.escape(fmt_date(q.get("createdAt")))
            img_url = q.get("questionImageUrl")

            # Image block
            if img_url:
                esc_url = html.escape(img_url)
                img_block = f'<div class="card-img-wrap"><img src="{esc_url}" alt="figure" onclick="openModal(\'{esc_url}\')" title="Click to zoom"></div>'
            else:
                img_block = ""

            # Answer + hint
            if has_ans:
                ans_block = f'<div class="card-section"><span class="section-label">Answer</span><div class="card-ans">{ans_text}</div></div>'
            else:
                ans_block = ""
            if has_hint:
                hint_block = f'<div class="card-section"><span class="section-label">Hint</span><div class="card-hint">{hint_text}</div></div>'
            else:
                hint_block = ""

            body_layout = "card-body-split" if img_url else "card-body-single"

            return (
                f'<div class="{card_cls}"'
                f' data-id="{rid}"'
                f' data-type="{html.escape(str(q.get("questionType") or "").lower())}"'
                f' data-diff="{html.escape(str(q.get("difficulty") or "").lower())}"'
                f' data-bloom="{html.escape(str(q.get("bloomLevel") or "").lower())}"'
                f' data-status="{html.escape(str(q.get("status") or "").lower())}"'
                f' data-answer="{"yes" if has_ans else "no"}"'
                f' data-dup="{"yes" if is_dup else "no"}"'
                f' data-text="{html.escape(qtext.lower()[:400])}"'
                f'>'
                # Top bar
                f'<div class="card-topbar">'
                f'  <span class="card-id badge badge-gray">#{rid}</span>'
                f'  {type_badge(q.get("questionType"))}'
                f'  {diff_badge(q.get("difficulty"))}'
                f'  <span class="badge badge-gray">{html.escape(str(q.get("bloomLevel") or ""))}</span>'
                f'  {status_badge(q.get("status"))}'
                f'  {dup_marker}'
                f'  <span class="card-ts"><i data-lucide="clock" style="width:11px;height:11px;"></i> {saved_fmt}</span>'
                f'</div>'
                # Body
                f'<div class="{body_layout}">'
                f'  <div class="card-text">{html.escape(qtext)}</div>'
                f'  {img_block}'
                f'</div>'
                # Answer / hint
                f'{ans_block}'
                f'{hint_block}'
                f'</div>'
            )
        q_cards = "".join(_card(q) for q in qs)
        has_dups = any(q.get("id") in dup_ids for q in qs)
        dup_warn = ' <span class="badge badge-red" style="font-size:0.65rem;">has duplicates</span>' if has_dups else ""
        batches_html_parts.append(f"""
        <div class="batch-block">
          <div class="batch-header">
            <div class="batch-meta">
              <i data-lucide="clock" style="width:13px;height:13px;flex-shrink:0;"></i>
              <span class="batch-time">{html.escape(fmt_minute(key))}</span>
              <span class="batch-sep">&mdash;</span>
              <span class="badge badge-gray">{len(qs)} question{'s' if len(qs) != 1 else ''}</span>
              {dup_warn}
            </div>
          </div>
          <div class="cards-grid">{q_cards}</div>
        </div>""")
    batches_html = "".join(batches_html_parts)

    # Duplicate cards
    dup_html = ""
    if dups:
        parts = []
        for text, items in sorted(dups.items(), key=lambda kv: -len(kv[1])):
            sample = items[0].get("questionText") or ""
            rows = "".join(
                f'<li>id={q.get("id")} · {html.escape(fmt_date(q.get("createdAt")))} · {html.escape(str(q.get("questionType","")))} · ch={q.get("chapterId")}</li>'
                for q in items
            )
            parts.append(
                f'<div class="card dup-card">'
                f'<div class="dup-count"><i data-lucide="copy" style="width:11px;height:11px;"></i>&nbsp;{len(items)} copies</div>'
                f'<div class="qtext">{html.escape(sample)}</div>'
                f'<ul class="meta-list">{rows}</ul></div>'
            )
        dup_html = "".join(parts)
    else:
        dup_html = '<p class="muted" style="padding:4px 0;">No exact duplicates found.</p>'

    chapter_label = f"chapter {chapterId}" if chapterId is not None else "all chapters"
    empty = "" if questions else f'<p class="muted">No questions found for course {courseId} ({chapter_label}).</p>'
    generated = _dt.now(_tz.utc).strftime("%Y-%m-%d %I:%M %p UTC")

    # Distinct values for filter chips
    def _vals(key):
        return sorted({str(q.get(key) or "").strip() for q in questions if (q.get(key) or "").strip()})
    types_vals = _vals("questionType")
    diff_vals  = _vals("difficulty")
    bloom_vals = _vals("bloomLevel")
    status_vals = _vals("status")

    def chip_row(group_id, values, label):
        chips = "".join(
            f'<span class="chip" data-group="{html.escape(group_id)}" data-val="{html.escape(v)}" onclick="toggleChip(this)">{html.escape(v)}</span>'
            for v in values
        )
        return (
            f'<span class="chip-label">{html.escape(label)}</span>'
            f'{chips}'
            f'<span class="chip-sep"></span>'
        ) if values else ""

    return f"""<style>
  .subtitle {{ color: #64748b; margin: 3px 0 20px; font-size: 0.85rem; }}
  h2 {{ font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin: 24px 0 10px; }}
  .btn-primary {{ background: #2563eb; color: white; border: none; }}
  .btn-primary:hover {{ background: #1d4ed8; }}
  .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin-bottom: 10px; }}
  .filter-row {{ display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }}
  .filter-row label {{ display: flex; flex-direction: column; gap: 3px; font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .filter-row input[type=number] {{ width: 100px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; font-family: inherit; -moz-appearance: textfield; appearance: textfield; }}
  .filter-row input[type=number]::-webkit-outer-spin-button,
  .filter-row input[type=number]::-webkit-inner-spin-button {{ -webkit-appearance: none; margin: 0; }}
  .filter-row input:focus {{ outline: none; border-color: #2563eb; }}
  .search-bar {{ width: 100%; padding: 7px 11px 7px 34px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; font-family: inherit; background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat 10px center; }}
  .search-bar:focus {{ outline: none; border-color: #2563eb; }}
  .chip-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; align-items: center; }}
  .chip-label {{ font-size: 0.68rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-right: 2px; }}
  .chip {{ padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; cursor: pointer; border: 1px solid #e2e8f0; background: #f8fafc; color: #475569; transition: background 0.1s, color 0.1s, border-color 0.1s; user-select: none; }}
  .chip:hover {{ background: #e2e8f0; }}
  .chip.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  .chip-sep {{ width: 1px; height: 18px; background: #e2e8f0; margin: 0 2px; }}
  .filter-count {{ font-size: 0.72rem; color: #64748b; margin-left: auto; }}
  .summary-strip {{ display: flex; flex-wrap: wrap; gap: 20px; }}
  .summary-strip .stat {{ display: flex; flex-direction: column; }}
  .summary-strip .stat-val {{ font-size: 1.5rem; font-weight: 700; line-height: 1.1; }}
  .summary-strip .stat-lbl {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .dup-card {{ border-left: 3px solid #dc2626; }}
  .dup-count {{ display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; font-weight: 700; color: #dc2626; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
  .qtext {{ font-size: 0.875rem; line-height: 1.7; margin-bottom: 10px; word-break: break-word; overflow-wrap: break-word; }}
  .meta-list {{ list-style: none; padding: 0; display: flex; flex-direction: column; gap: 4px; }}
  .meta-list li {{ font-size: 0.75rem; color: #64748b; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 4px 8px; font-family: monospace; }}
  .badge {{ padding: 2px 7px; border-radius: 9999px; font-size: 0.65rem; font-weight: 600; white-space: nowrap; }}
  .badge-blue {{ background: #dbeafe; color: #1d4ed8; }}
  .badge-green {{ background: #dcfce7; color: #166534; }}
  .badge-gray {{ background: #f1f5f9; color: #64748b; }}
  .badge-red {{ background: #fee2e2; color: #dc2626; }}
  .batch-block {{ margin-bottom: 24px; }}
  .batch-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }}
  .batch-meta {{ display: flex; align-items: center; gap: 7px; font-size: 0.78rem; flex-wrap: wrap; }}
  .batch-time {{ font-weight: 600; color: #1e293b; }}
  .batch-sep {{ color: #cbd5e1; }}
  .cards-grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .q-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }}
  .dup-card-q {{ border-left: 3px solid #f59e0b; }}
  .card-topbar {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; padding: 10px 14px; border-bottom: 1px solid #f1f5f9; background: #f8fafc; }}
  .card-id {{ font-family: monospace; }}
  .card-ts {{ margin-left: auto; display: inline-flex; align-items: center; gap: 4px; font-size: 0.7rem; color: #94a3b8; white-space: nowrap; }}
  .dup-marker-badge {{ display: inline-flex; align-items: center; gap: 3px; font-size: 0.65rem; font-weight: 700; color: #d97706; background: #fef3c7; border: 1px solid #fde68a; border-radius: 9999px; padding: 1px 7px; }}
  .card-body-split {{ display: grid; grid-template-columns: 1fr auto; gap: 16px; padding: 16px 14px; align-items: start; }}
  .card-body-single {{ padding: 16px 14px; }}
  .card-text {{ font-size: 0.9rem; line-height: 1.75; color: #1e293b; word-break: break-word; overflow-wrap: break-word; }}
  .card-img-wrap {{ flex-shrink: 0; }}
  .card-img-wrap img {{ max-width: 220px; max-height: 180px; border: 1px solid #e2e8f0; border-radius: 7px; display: block; cursor: zoom-in; object-fit: contain; }}
  .card-section {{ padding: 10px 14px; border-top: 1px solid #f1f5f9; }}
  .section-label {{ display: block; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; margin-bottom: 4px; }}
  .card-ans {{ font-size: 0.82rem; line-height: 1.65; color: #1e293b; white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word; }}
  .card-hint {{ font-size: 0.82rem; line-height: 1.65; color: #64748b; font-style: italic; white-space: pre-wrap; word-break: break-word; overflow-wrap: break-word; }}
  .img-modal {{ position: fixed; inset: 0; background: rgba(15,23,42,0.8); display: none; align-items: center; justify-content: center; z-index: 300; padding: 18px; }}
  .img-modal.open {{ display: flex; }}
  .img-modal img {{ max-width: min(1100px,96vw); max-height: 90vh; border-radius: 8px; box-shadow: 0 16px 40px rgba(15,23,42,0.45); background: white; object-fit: contain; }}
  .img-modal-close {{ position: absolute; top: 14px; right: 14px; border: 1px solid #94a3b8; background: #0f172a; color: white; border-radius: 6px; font-size: 0.78rem; padding: 6px 10px; cursor: pointer; }}
  .muted {{ color: #94a3b8; font-size: 0.78rem; }}
  /* Dark mode */
  body.dark .card {{ background: #1e293b; border-color: #334155; }}
  body.dark .subtitle {{ color: #64748b; }}
  body.dark h2 {{ color: #475569; }}
  body.dark .filter-row label {{ color: #64748b; }}
  body.dark .filter-row input[type=number] {{ background: #0f172a; border-color: #334155; color: #e2e8f0; }}
  body.dark .filter-row input:focus {{ border-color: #2563eb; }}
  body.dark .search-bar {{ background-color: #0f172a; border-color: #334155; color: #e2e8f0; }}
  body.dark .search-bar:focus {{ border-color: #2563eb; }}
  body.dark .chip {{ background: #1e293b; border-color: #334155; color: #94a3b8; }}
  body.dark .chip:hover {{ background: #334155; color: #e2e8f0; }}
  body.dark .chip.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  body.dark .chip-sep {{ background: #334155; }}
  body.dark .filter-count {{ color: #475569; }}
  body.dark .summary-strip .stat-lbl {{ color: #475569; }}
  body.dark .dup-card {{ border-left-color: #f87171; }}
  body.dark .dup-count {{ color: #f87171; }}
  body.dark .meta-list li {{ background: #0f172a; border-color: #334155; color: #94a3b8; }}
  body.dark .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
  body.dark .badge-green {{ background: #14532d; color: #86efac; }}
  body.dark .badge-gray {{ background: #334155; color: #94a3b8; }}
  body.dark .badge-red {{ background: #450a0a; color: #f87171; }}
  body.dark .muted {{ color: #475569; }}
  body.dark .batch-time {{ color: #e2e8f0; }}
  body.dark .batch-sep {{ color: #334155; }}
  body.dark .q-card {{ background: #1e293b; border-color: #334155; }}
  body.dark .dup-card-q {{ border-left-color: #f59e0b; }}
  body.dark .card-topbar {{ background: #0f172a; border-bottom-color: #334155; }}
  body.dark .card-text {{ color: #e2e8f0; }}
  body.dark .card-section {{ border-top-color: #334155; }}
  body.dark .card-ans {{ color: #e2e8f0; }}
  body.dark .card-hint {{ color: #94a3b8; }}
  body.dark .card-img-wrap img {{ border-color: #334155; }}
</style>

<p class="subtitle">Course {courseId} &mdash; {chapter_label} &mdash; {len(questions)} of {total} questions &mdash; generated {generated}</p>

<!-- Filter card -->
<div class="card" style="margin-bottom:16px;">
  <form method="get" action="/latest-questions">
    <div class="filter-row">
      <label>Course ID
        <input type="number" name="courseId" value="{courseId}" required>
      </label>
      <label>Chapter ID (optional)
        <input type="number" name="chapterId" value="{chapterId if chapterId is not None else ''}">
      </label>
      <label>Limit
        <input type="number" name="limit" value="{limit}" min="1" max="500">
      </label>
      <button type="submit" class="btn btn-primary btn-sm" style="margin-bottom:1px;">
        <i data-lucide="refresh-cw" style="width:13px;height:13px;"></i>
        Refresh
      </button>
    </div>
  </form>
</div>

<!-- Summary -->
<div class="card" style="margin-bottom:16px;">
  <div class="summary-strip">
    <div class="stat"><span class="stat-val">{len(questions)}</span><span class="stat-lbl">Showing</span></div>
    <div class="stat"><span class="stat-val">{total}</span><span class="stat-lbl">Total</span></div>
    <div class="stat"><span class="stat-val">{len(dups)}</span><span class="stat-lbl">Dup groups</span></div>
    <div class="stat"><span class="stat-val">{sum(len(v) for v in dups.values())}</span><span class="stat-lbl">Dup rows</span></div>
  </div>
</div>

{empty}

<!-- Live search + filter chips -->
<div class="card" style="margin-bottom:16px;">
  <input class="search-bar" id="q-search" type="text" placeholder="Search by ID or question text…" oninput="applyFilters()">
  <div class="chip-row" id="chip-area">
    {chip_row("type", types_vals, "Type")}
    {chip_row("difficulty", diff_vals, "Difficulty")}
    {chip_row("bloom", bloom_vals, "Bloom")}
    {chip_row("status", status_vals, "Status")}
    <span class="chip" data-group="answer" data-val="yes" onclick="toggleChip(this)">Has answer</span>
    <span class="chip" data-group="answer" data-val="no" onclick="toggleChip(this)">No answer</span>
    <span class="chip-sep"></span>
    <span class="chip" data-group="dup" data-val="yes" onclick="toggleChip(this)">Duplicates only</span>
    <span class="filter-count" id="filter-count"></span>
  </div>
</div>

<!-- Duplicates -->
<h2><i data-lucide="copy" style="width:12px;height:12px;display:inline;vertical-align:middle;margin-right:4px;"></i>Duplicates (exact text)</h2>
{dup_html}

<!-- Batch groups -->
<h2><i data-lucide="layers" style="width:12px;height:12px;display:inline;vertical-align:middle;margin-right:4px;"></i>Save batches (grouped by minute)</h2>
{batches_html}

<!-- Image zoom modal (appended to body by script below) -->
<template id="modal-tpl">
  <div class="img-modal" id="img-modal" onclick="closeModal()">
    <button class="img-modal-close" onclick="closeModal()">Close</button>
    <img id="img-modal-src" src="" alt="figure">
  </div>
</template>

<script>
  // Mount modal onto body
  (function() {{
    var tpl = document.getElementById('modal-tpl');
    if (tpl) document.body.appendChild(tpl.content.cloneNode(true));
  }})();

  // Image zoom (global so inline onclick works)
  window.openModal = function(src) {{
    document.getElementById('img-modal-src').src = src;
    document.getElementById('img-modal').classList.add('open');
  }};
  window.closeModal = function() {{
    document.getElementById('img-modal').classList.remove('open');
    document.getElementById('img-modal-src').src = '';
  }};
  document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') window.closeModal(); }});

  // Live search + filter chips
  var activeChips = {{}};

  window.toggleChip = function(el) {{
    var group = el.dataset.group;
    var val   = el.dataset.val;
    if (!activeChips[group]) activeChips[group] = new Set();
    if (activeChips[group].has(val)) {{
      activeChips[group].delete(val);
      el.classList.remove('active');
    }} else {{
      if (group === 'answer' || group === 'dup') {{
        document.querySelectorAll('.chip[data-group="' + group + '"]').forEach(function(c) {{
          c.classList.remove('active');
        }});
        activeChips[group] = new Set();
      }}
      activeChips[group].add(val);
      el.classList.add('active');
    }}
    window.applyFilters();
  }};

  window.applyFilters = function() {{
    var search = (document.getElementById('q-search').value || '').toLowerCase().trim();
    var cards = document.querySelectorAll('.q-card');
    var visible = 0;
    cards.forEach(function(card) {{
      var idMatch   = !search || card.dataset.id.includes(search);
      var textMatch = !search || card.dataset.text.includes(search);
      if (!idMatch && !textMatch) {{ card.style.display = 'none'; return; }}
      for (var group in activeChips) {{
        if (!activeChips[group].size) continue;
        if (!activeChips[group].has(card.dataset[group])) {{ card.style.display = 'none'; return; }}
      }}
      card.style.display = '';
      visible++;
    }});
    document.querySelectorAll('.batch-block').forEach(function(block) {{
      var anyVisible = Array.from(block.querySelectorAll('.q-card')).some(function(c) {{
        return c.style.display !== 'none';
      }});
      block.style.display = anyVisible ? '' : 'none';
    }});
    var fc = document.getElementById('filter-count');
    if (fc) fc.textContent = (search || Object.values(activeChips).some(function(s){{return s.size;}}))
      ? visible + ' match' + (visible !== 1 ? 'es' : '') : '';
  }};
</script>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)), reload=True)
