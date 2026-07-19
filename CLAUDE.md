# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Chinese-language knowledge base for AI/ML algorithm engineer interview preparation — a curated collection of Markdown files covering AIGC, LLMs, deep learning, computer vision, model deployment, and related topics. There is no code, no build system, and no test suite. All work is editing and organizing Markdown documentation.

## Repository architecture

Top-level directories are self-contained knowledge domains. Each has its own `readme.md` as a table of contents and its own `imgs/` directory for local image assets.

**"精华版" (essential edition) vs full edition**: four topics — 深度学习基础, 大模型基础, AI视频基础, 模型部署基础 — have both a full directory and a `（精华版）` variant. The 精华版 is a restructuring/simplification of the full version into fewer, longer files with a more consistent format. When updating one variant, check whether the other needs the same update.

**Numbered prefixes** (e.g. `01_`, `02_`) on filenames control ordering within a directory. These are significant — don't renumber casually.

## Content conventions

The `AI视频基础（精华版）/Markdown格式范式.md` file demonstrates the standard format for Q&A-style content:

- Each question gets an `<h1 id="...">` header with a stable anchor ID
- Difficulty and frequency ratings use star notation: `**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**`
- Questions are organized under chapter-level `#` and `##` headings
- Images referenced with relative paths into the local `imgs/` subdirectory

All content is in Chinese. Terminology should be consistent across files — prefer standard Chinese AI/ML terms (e.g., 扩散模型 not Diffusion Model, 大模型 not LLM, though mixed usage appears in places).

## Common operations

- **Editing content**: Use `Edit` tool directly on `.md` files. Files can be very long (2000+ lines) — use `Read` with offset/limit when reading large files.
- **Finding content**: Use `Grep` across all `*.md` files to find where a term or concept is discussed before adding duplicate coverage.
- **Adding images**: Place images in the topic's local `imgs/` directory and reference with a relative path like `imgs/filename.png`.
- **Adding a new question/topic**: Follow the established format in the file you're editing. Add to the nearest relevant section rather than creating a new file unless the topic is genuinely new.
- **Commits**: Recent commits follow a simple Chinese-language message style (e.g., "推理部署综述更新"). Keep messages short and descriptive of the content change.
