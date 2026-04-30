---
name: publish-mlbot-blog-post
description: Use when turning ml-examples research notes, reports, figures, or experiment outputs into a Markdown post for mlbot.blog, especially when the user says publish, blog, post, write up, summarize results, or put this on the blog. Do not use for ordinary code changes or non-blog documentation.
---

# Publish mlbot.blog Post

## Purpose

Publish research work from `/Users/djwurtz/proj/ml-examples` to the Hugo blog in `/Users/djwurtz/proj/mlbot-blog`.

The research source is the public GitHub repo `dwrtz/ml-examples` at `https://github.com/dwrtz/ml-examples`. The blog source is the public GitHub repo `dwrtz/mlbot-blog` at `https://github.com/dwrtz/mlbot-blog`, and the production site is `https://mlbot.blog/`.

The agent has permission to commit, push, and publish blog posts in the blog repo when the user asks to publish or put something on the blog. Do not request additional confirmation just to push `main` and `live`; do ask if the source material seems sensitive, private, or scientifically ambiguous.

The output should be a reviewable Hugo leaf bundle under `content/posts/` with clear prose, useful supporting assets, valid front matter, and no sensitive data.

## Workflow

1. Establish the source material.
   - Inspect the relevant files in `ml-examples`: reports in `docs/`, notes, scripts, configs, plots, or `outputs/`.
   - Prefer committed or durable source artifacts over transient logs.
   - Do not commit or revert changes in `ml-examples` unless the user explicitly asks.

2. Decide publication state.
   - If the user asks to publish, deploy, make public, or put it on the blog: create a publishable post with `draft: false`, `params.status: "published"`, and `params.reviewed_by: "dwrtz"` unless the user gives a different reviewer id.
   - If the user asks for a draft or review copy: use `draft: true`, `params.status: "review"`, and `params.reviewed_by: ""`.

3. Create a Hugo leaf bundle in `/Users/djwurtz/proj/mlbot-blog`:

   ```text
   content/posts/YYYY-MM-DD-short-kebab-slug/
   ├── index.md
   ├── cover.png
   ├── plots/
   └── artifacts/
   ```

   The public URL must be `/posts/<slug>/`; keep the date in the directory name only.

4. Write front matter exactly in YAML form:

   ```yaml
   ---
   title: "Concise Title"
   description: "One sentence, ideally 80-200 characters, suitable for search and social previews."
   date: YYYY-MM-DDTHH:MM:SS+05:30
   draft: false
   slug: "concise-kebab-slug"
   tags:
     - ai-research
   series: []
   images:
     - cover.png
   params:
     author: "mlbot"
     math: true
     generated_by: "ai-agent"
     reviewed_by: "dwrtz"
     status: "published"
     summary_kind: "research-note"
     canonical: "https://mlbot.blog/posts/concise-kebab-slug/"
   ---
   ```

   Use `params.math: true` when the post contains LaTeX delimiters. Use `summary_kind` from `research-note`, `experiment-report`, `essay`, `announcement`, or `link-note`.

5. Write the article body.
   - Use conservative Markdown: headings, paragraphs, lists, fenced code blocks, and LaTeX math.
   - Do not write raw HTML, iframes, remote scripts, tracking pixels, or analytics.
   - Keep claims grounded in the source artifacts; explicitly label speculation or interpretation.
   - Include enough methodological detail to make the result auditable: command/config names, seeds, metrics, baselines, caveats, and artifact paths when relevant.
   - Enhance the post when the source material supports it:
     - figures, plots, or diagrams that make the result easier to understand;
     - Markdown tables for compact comparisons, metrics, ablations, or run summaries;
     - small code snippets that clarify the method or reproduce a key command;
     - LaTeX equations with surrounding prose that explains the math, not just displays it;
     - links to relevant external sources, docs, papers, or literature;
     - links to relevant `ml-examples` commits, files, scripts, configs, or result reports in `https://github.com/dwrtz/ml-examples`;
     - links to related `mlbot.blog` posts, especially for sequences, progress updates, or follow-ups;
     - specific, reusable tags that connect the post to adjacent work.
   - Do not duplicate the cover image inside the Markdown body. The post template renders `images[0]` as the cover.

6. Assets and artifacts.
   - Copy only non-sensitive, publication-worthy assets into the post bundle.
   - Put plots under `plots/` and downloadable data under `artifacts/`.
   - Prefer PNG or SVG plots that render directly in the static site.
   - Never include credentials, private prompts, private datasets, unpublished sensitive results, or large raw outputs.

7. Validate from `/Users/djwurtz/proj/mlbot-blog`.
   - Always run `npm run validate`.
   - For publishable posts, run `npm run validate:publish`.
   - Run `PATH="$PWD/.cache/bin:$PATH" npm run check` when local Hugo is available; otherwise run `npm run check` and report if Hugo is missing.

8. Publish.
   - For a public post, commit changes in `/Users/djwurtz/proj/mlbot-blog`, push `main`, then push `main:live`.
   - Watch the GitHub Actions `Publish` workflow for `live` and verify it succeeds.
   - Verify the final URL returns `200` and the post appears on `/posts/`.
   - For a draft post, leave it committed or uncommitted according to the user’s request; do not push to `live`.

## Quality Bar

- The post should read like a short research note, not a changelog dump.
- The title and description should be understandable without reading the code.
- Math must use standard delimiters (`\( ... \)`, `\[ ... \]`, or `$$ ... $$`).
- Code must use fenced blocks with language tags.
- Tags should be useful for navigation, not one-off labels unless the topic is truly unique.
- Prefer at least one relevant internal or external link when the post builds on prior work or literature.
- Prefer stable GitHub links to important source files and, when useful, exact commits that produced or changed the result.
- The post must pass the blog validator before being described as ready.
