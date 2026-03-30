# Vendored OpenAI CLIP

This directory contains vendored code from the OpenAI CLIP repository:

- Source: https://github.com/openai/CLIP
- Upstream license: MIT

This copy is included inside `enczoo` so `CLIPResNet50` works without requiring
users to install a separate `clip` package.

If you modify files in this directory, keep the changes minimal and preserve
clear provenance back to the upstream project so future updates remain
tractable.

Recommended maintenance practice:

- record the upstream commit or release used for vendoring
- document any local modifications in commit messages or file headers
- preserve upstream license notices alongside the vendored code
