---
title: EduVerse Question Generator
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# EduVerse Question Generator

Extracts and structures questions from textbook PDFs using Gemini AI, then saves them directly to the EduVerse question bank.

## Setup

Set the following environment variables in Space Settings → Variables:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key |
| `EDUVERSE_API_URL` | EduVerse backend URL |
| `EDUVERSE_EMAIL` | Instructor email |
| `EDUVERSE_PASSWORD` | Instructor password |
