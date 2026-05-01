import asyncio
import base64
import io
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


def _active_pdf() -> Optional[Path]:
    """Return the PDF to use: user-uploaded first, then the bundled textbook."""
    if SAVED_PDF.exists():
        return SAVED_PDF
    if BUNDLED_PDF.exists():
        return BUNDLED_PDF
    return None

GEMINI_PROMPT = """\
You are extracting end-of-chapter problems from a Control Systems textbook (Modern Control Systems, 12th edition).

The text was extracted via OCR from a scanned PDF and may contain artifacts, merged columns, and garbled spacing. Identify every distinct problem.

Return a JSON array where each object has exactly these fields:
- "question_number": string — the problem label as it appears (e.g. "P1.1", "AP3.4", "DP5.1", "CP2.1")
- "question": string — complete question text, cleaned of OCR errors. Remove figure captions and page headers. Fix broken words from column wrapping. Format all math using LaTeX: inline with $...$ (e.g. $x_1$, $\\omega_n$, $K > 0$), display with $$...$$.
- "page": number — page where the problem starts (use the --- PAGE N --- markers)
- "has_diagram": boolean — true if the question references a Figure, asks to sketch a diagram or block diagram, or involves a circuit
- "figure_reference": string or null — specific figure label when present (e.g. "Figure 13.18", "Fig. P1.2"), otherwise null
- "exam_ready": boolean — true if suitable for a university exam. Mark false for: pure discussion/describe questions, problems entirely dependent on a figure students won't have, or questions too open-ended to grade objectively.
- "exam_notes": string or null — if exam_ready is false, a brief reason (e.g. "discussion only", "requires Figure P1.2", "open-ended design"). If exam_ready is true, use null.

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
    onload="onKatexReady()"></script>
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
    .config-bar input:focus { outline: none; border-color: #2563eb; }
    .config-sep { color: #94a3b8; font-size: 0.85rem; }
    #extract-status { font-size: 0.78rem; color: #64748b; }
    /* Spinner (inline) */
    .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.4); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    /* PDF picker */
    .pdf-picker-btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px; background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 6px; font-size: 0.78rem; font-weight: 600; color: #475569; cursor: pointer; transition: border-color 0.15s, background 0.15s, color 0.15s; user-select: none; }
    .pdf-picker-btn:hover { border-color: #2563eb; color: #2563eb; background: #eff6ff; }
    .pdf-ready-pill { display: none; align-items: center; gap: 7px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 5px 11px; font-size: 0.78rem; color: #166534; max-width: 340px; }
    .pdf-ready-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
    .pdf-pending-pill { display: none; align-items: center; gap: 7px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 5px 11px; font-size: 0.78rem; color: #92400e; max-width: 380px; }
    .pdf-pending-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px; }
    .pdf-change-link { font-size: 0.7rem; color: #64748b; text-decoration: underline; cursor: pointer; flex-shrink: 0; }
    .pdf-change-link:hover { color: #2563eb; }
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
    /* Problem cards */
    .q-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; transition: border-color 0.15s; cursor: default; }
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
    .exam-toggle { display: inline-flex; align-items: center; gap: 5px; font-size: 0.73rem; color: #64748b; cursor: pointer; user-select: none; margin-left: auto; }
    .exam-toggle input[type=checkbox] { cursor: pointer; }
    .q-notes { font-size: 0.71rem; color: #94a3b8; margin-top: 5px; font-style: italic; }
    .chips-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .img-chip { display: inline-flex; align-items: center; gap: 5px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 3px 8px 3px 3px; font-size: 0.72rem; color: #475569; }
    .img-chip img { width: 56px; height: 56px; object-fit: cover; border-radius: 4px; cursor: zoom-in; border: 1px solid #cbd5e1; }
    .paste-target-pill { margin-left: auto; font-size: 0.68rem; color: #1d4ed8; background: #dbeafe; border: 1px solid #93c5fd; border-radius: 9999px; padding: 2px 7px; display: none; }
    .q-card.active-card .paste-target-pill { display: inline-flex; align-items: center; }
    .img-chip-remove { background: none; border: none; cursor: pointer; color: #94a3b8; font-size: 1rem; line-height: 1; padding: 0; }
    .img-chip-remove:hover { color: #dc2626; }
    /* Export bar */
    .export-bar { position: fixed; bottom: 0; left: 0; right: 0; background: white; border-top: 1px solid #e2e8f0; padding: 10px 24px; display: none; justify-content: space-between; align-items: center; z-index: 100; }
    .export-bar-left { display: flex; align-items: center; gap: 14px; font-size: 0.82rem; color: #64748b; }
    .paste-hint { font-size: 0.75rem; color: #2563eb; background: #eff6ff; padding: 2px 8px; border-radius: 4px; }
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
    body.dark .pdf-picker-btn { background: #1e293b; border-color: #475569; color: #94a3b8; }
    body.dark .pdf-picker-btn:hover { border-color: #2563eb; color: #93c5fd; background: #1e3a5f; }
    body.dark .pdf-change-link { color: #64748b; }
    body.dark .pdf-change-link:hover { color: #93c5fd; }
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
    <button class="dm-toggle" id="dm-toggle-btn" onclick="toggleDark()" title="Toggle dark mode" aria-label="Toggle dark mode">
      <svg id="dm-icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <svg id="dm-icon-sun"  width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    </button>
  </div>
  <p class="subtitle">Extract text &rarr; Gemini structures problems &rarr; review &amp; export.</p>

  <div id="error-box" class="error-box"></div>

  <!-- ── Config bar ── -->
  <div class="card" style="margin-bottom:16px;">
    <div class="config-bar">
      <!-- PDF picker -->
      <input type="file" id="pdf-file-input" accept=".pdf" style="display:none;" onchange="handlePdfSelect(this)">
      <label id="pdf-picker-btn" class="pdf-picker-btn" for="pdf-file-input">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Choose PDF
      </label>
      <div id="pdf-ready-pill" class="pdf-ready-pill">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        <span id="pdf-ready-name" class="pdf-ready-name"></span>
        <span class="pdf-change-link" onclick="document.getElementById('pdf-file-input').click()">Change</span>
      </div>
      <div id="pdf-pending-pill" class="pdf-pending-pill">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span id="pdf-pending-name" class="pdf-pending-name"></span>
        <span style="color:#a16207;font-size:0.68rem;flex-shrink:0;">uploads on Extract</span>
        <span class="pdf-change-link" onclick="document.getElementById('pdf-file-input').click()">Change</span>
      </div>
      <!-- Divider -->
      <span class="config-sep">|</span>
      <!-- Pages -->
      <span style="font-size:0.78rem;color:#64748b;">Pages</span>
      <input id="tb-start" type="number" min="1" placeholder="Start">
      <span class="config-sep">&ndash;</span>
      <input id="tb-end" type="number" min="1" placeholder="End">
      <!-- Extract -->
      <button id="extract-btn" class="btn btn-primary btn-sm" onclick="startExtract()">Extract</button>
      <span id="extract-status"></span>
    </div>
  </div>

  <!-- ── Workflow (always available) ── -->
  <div id="workflow-section">
    <div class="workflow-grid">
      <div class="card">
        <div class="step-label"><span class="step-num">1</span> Copy &amp; paste into <a href="https://gemini.google.com" target="_blank" style="color:#2563eb;">gemini.google.com</a></div>
        <div class="info-box">Copies the full prompt + your extracted text in one click.</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button id="copy-btn" class="btn btn-primary btn-sm" onclick="copyForGemini()">Copy prompt + text</button>
          <button id="download-raw-btn" class="btn btn-secondary btn-sm" onclick="downloadRawText()">Download .txt</button>
        </div>
      </div>
      <div class="card">
        <div class="step-label"><span class="step-num">2</span> Paste Gemini&rsquo;s JSON response</div>
        <textarea id="json-paste" class="paste-area" placeholder="Paste here (with or without the ```json fences)..."></textarea>
        <button id="load-json-btn" class="btn btn-success btn-sm" onclick="loadFromJSON()">Load Problems &rarr;</button>
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

<!-- Export bar -->
<div id="export-bar" class="export-bar">
  <div class="export-bar-left">
    <span id="q-count"></span>
    <span id="paste-hint" class="paste-hint" style="display:none;">Ctrl+V &rarr; <span id="paste-target-label"></span></span>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <input id="chapter-export" type="number" min="1" placeholder="Chapter" style="width:88px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.78rem;">
    <input id="teammate-export" type="text" placeholder="Teammate (optional)" style="width:160px;padding:6px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:0.78rem;">
    <button class="btn btn-secondary btn-sm" onclick="exportChapterZip(this)">Export chapter ZIP</button>
    <button class="btn btn-success btn-sm"   onclick="exportJSON()">Export JSON</button>
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

  // ── PDF state ─────────────────────────────────────────────────────────────

  function initPdfState() {
    fetch('/api/pdf-info').then(function(r) { return r.json(); }).then(function(d) {
      if (d.saved && d.name) { showPdfReady(d.name); }
    }).catch(function() {});
  }

  function handlePdfSelect(input) {
    var file = input.files && input.files[0];
    if (!file) return;
    pendingPdfFile = file;
    document.getElementById('pdf-picker-btn').style.display = 'none';
    document.getElementById('pdf-ready-pill').style.display  = 'none';
    var pill = document.getElementById('pdf-pending-pill');
    pill.style.display = 'inline-flex';
    document.getElementById('pdf-pending-name').textContent = file.name;
    input.value = '';
  }

  function showPdfReady(name) {
    pendingPdfFile = null;
    document.getElementById('pdf-picker-btn').style.display  = 'none';
    document.getElementById('pdf-pending-pill').style.display = 'none';
    var pill = document.getElementById('pdf-ready-pill');
    pill.style.display = 'inline-flex';
    document.getElementById('pdf-ready-name').textContent = name;
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

  window.addEventListener('DOMContentLoaded', function() {
    initPdfState();
    try {
      var saved = localStorage.getItem('dm');
      var prefersDark = saved !== null ? saved === '1' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyDark(prefersDark);
    } catch(e) {}
  });

  var KATEX_OPTS = {
    delimiters: [
      { left: '$$', right: '$$', display: true  },
      { left: '$',  right: '$',  display: false }
    ],
    throwOnError: false
  };

  // ── KaTeX ─────────────────────────────────────────────────────────────────

  function onKatexReady() {
    katexReady = true;
    document.querySelectorAll('.q-rendered').forEach(function(el) {
      renderMathInElement(el, KATEX_OPTS);
    });
  }

  function renderMath(el) {
    if (katexReady && window.renderMathInElement) {
      renderMathInElement(el, KATEX_OPTS);
    }
  }

  // ── Extract ───────────────────────────────────────────────────────────────

  async function startExtract() {
    hideError();
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
        '\\u2713 ' + (data.char_count || 0).toLocaleString() + ' chars \\u00b7 ' + imgs.length + ' images';
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
    document.getElementById('images-arrow').textContent = '\\u25ba';
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
    arrow.textContent     = open ? '\\u25ba' : '\\u25bc';
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
          question_number: String(q.question_number || ''),
          question:        String(q.question || ''),
          page:            q.page || null,
          has_diagram:     Boolean(q.has_diagram),
          figure_reference: figureReference || null,
          exam_ready:      q.exam_ready !== false,
          exam_notes:      q.exam_notes || null,
          answer:          q.answer || null,
          images:          [],
        };
      });
      var ready = problems.filter(function(q) { return q.exam_ready; }).length;
      document.getElementById('review-meta').textContent =
        problems.length + ' problems loaded \\u00b7 ' + ready + ' exam ready';
      updateFilterCounts();
      renderProblems();
      var rs = document.getElementById('review-section');
      rs.style.display = 'block';
      document.getElementById('export-bar').style.display = 'flex';
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

  var TYPE_LABELS = {
    AP: 'Advanced Problems',
    P:  'Problems',
    E:  'Exercises',
    DP: 'Design Problems',
    CP: 'Computer Problems',
    EP: 'Extension Problems',
  };

  var TYPE_ORDER = ['AP', 'P', 'E', 'DP', 'CP', 'EP', 'OTHER'];

  function getQuestionTypeCode(questionNumber) {
    var qn = String(questionNumber || '').trim().toUpperCase();
    var m = qn.match(/^([A-Z]+)\s*\d/);
    if (!m) return 'OTHER';
    return m[1];
  }

  function getQuestionTypeLabel(typeCode) {
    if (TYPE_LABELS[typeCode]) return TYPE_LABELS[typeCode];
    return typeCode === 'OTHER' ? 'Other' : (typeCode + ' Questions');
  }

  function detectFigureReference(text) {
    var src = String(text || '');
    var m = src.match(/\\b(?:Figure|Fig\\.?)\\s*([A-Za-z]?\\d+(?:\\.\\d+)*)\\b/i);
    if (!m) return null;
    return 'Figure ' + m[1];
  }

  function getSortedTypeCodes(grouped) {
    var known = TYPE_ORDER.filter(function(code) { return grouped[code] && grouped[code].length; });
    var unknown = Object.keys(grouped).filter(function(code) {
      return TYPE_ORDER.indexOf(code) === -1 && grouped[code] && grouped[code].length;
    }).sort();
    return known.concat(unknown);
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
      var typeCode = getQuestionTypeCode(q.question_number);
      if (!grouped[typeCode]) grouped[typeCode] = [];
      grouped[typeCode].push({ question: q, index: idx });
    });

    getSortedTypeCodes(grouped).forEach(function(typeCode) {
      var section = document.createElement('section');
      section.className = 'q-section';
      section.dataset.type = typeCode;

      var title = document.createElement('div');
      title.className = 'q-section-title';
      title.textContent = getQuestionTypeLabel(typeCode) + ' (' + grouped[typeCode].length + ')';
      section.appendChild(title);

      grouped[typeCode].forEach(function(item) {
        section.appendChild(buildCard(item.question, item.index));
      });
      list.appendChild(section);
    });

    applyFilter();
  }

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
    if (q.figure_reference || q.has_diagram) {
      var dg = document.createElement('span'); dg.className = 'badge badge-warn';
      dg.textContent = '\\u26a0 ' + (q.figure_reference || 'diagram'); header.appendChild(dg);
    }
    var eb = document.createElement('span');
    eb.id = 'exam-badge-' + idx;
    eb.className = 'badge ' + (q.exam_ready ? 'badge-ready' : 'badge-skip');
    eb.textContent = q.exam_ready ? '\\u2713 exam ready' : '\\u00d7 skip';
    header.appendChild(eb);
    var targetPill = document.createElement('span');
    targetPill.className = 'paste-target-pill';
    targetPill.textContent = 'Ctrl+V target';
    header.appendChild(targetPill);
    var sp = document.createElement('div'); sp.style.flex = '1'; header.appendChild(sp);
    var del = document.createElement('button'); del.className = 'btn btn-danger btn-sm'; del.textContent = 'Delete';
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

    var editBtn = document.createElement('button'); editBtn.className = 'btn btn-secondary btn-xs'; editBtn.textContent = 'Edit';
    var doneBtn = document.createElement('button'); doneBtn.className = 'btn btn-primary btn-xs'; doneBtn.textContent = 'Done'; doneBtn.style.display = 'none';
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
    controls.appendChild(editBtn); controls.appendChild(doneBtn);

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
    var attachBtn = document.createElement('button'); attachBtn.className = 'btn btn-secondary btn-xs'; attachBtn.textContent = '+ Attach diagram';
    attachBtn.addEventListener('click', function(e) { e.stopPropagation(); fi.click(); });
    controls.appendChild(attachBtn);

    // Exam ready toggle
    var examLabel = document.createElement('label'); examLabel.className = 'exam-toggle';
    var examCb = document.createElement('input'); examCb.type = 'checkbox'; examCb.checked = Boolean(q.exam_ready);
    examCb.addEventListener('change', (function(i, cardEl) {
      return function(e) {
        problems[i].exam_ready = e.target.checked;
        cardEl.dataset.examReady = e.target.checked ? '1' : '0';
        var badge = document.getElementById('exam-badge-' + i);
        if (badge) { badge.className = 'badge ' + (e.target.checked ? 'badge-ready' : 'badge-skip'); badge.textContent = e.target.checked ? '\\u2713 exam ready' : '\\u00d7 skip'; }
        applyFilter(); updateFilterCounts();
      };
    })(idx, card));
    var examLbl = document.createElement('span'); examLbl.textContent = 'Exam ready';
    examLabel.appendChild(examCb); examLabel.appendChild(examLbl);
    controls.appendChild(examLabel);
    card.appendChild(controls);

    // Notes
    if (q.exam_notes) {
      var notes = document.createElement('div'); notes.className = 'q-notes';
      notes.textContent = 'Note: ' + q.exam_notes; card.appendChild(notes);
    }

    // Image chips
    var chipsRow = document.createElement('div'); chipsRow.className = 'chips-row';
    chipsRow.id = 'chips-' + idx;
    updateChips(idx, chipsRow); card.appendChild(chipsRow);

    return card;
  }

  function updateChips(idx, chipsRow) {
    chipsRow.innerHTML = '';
    (problems[idx].images || []).forEach(function(img, ii) {
      var chip = document.createElement('div'); chip.className = 'img-chip';
      var thumb = document.createElement('img'); thumb.src = img.data; thumb.title = img.name;
      thumb.addEventListener('click', function() { openImageModal(img.data); });
      var name = document.createElement('span'); name.textContent = img.name || 'image';
      name.style.cssText = 'max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
      var rm = document.createElement('button'); rm.className = 'img-chip-remove'; rm.textContent = '\\u00d7';
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
    var m = qn.match(/^[A-Z]+(\d+)\./);
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


@app.post("/api/parse-json")
async def parse_json_route(request: Request):
    raw = (await request.body()).decode("utf-8").strip()
    m = re.search(r"```(?:json)?\n([\s\S]*?)\n?```", raw)
    if m:
        raw = m.group(1).strip()
    # Fix unescaped LaTeX backslashes like \le, \ge, \alpha that Gemini forgets to escape.
    # Match every \X pair atomically: if X is a valid JSON escape char, leave it alone;
    # otherwise double the backslash. This prevents the second \ of \\ from being re-matched.
    fixed = re.sub(
        r'\\(.)',
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
                "answer": q.get("answer"),
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


@app.get("/output/{filename}")
def serve_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
