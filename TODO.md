# TODO

## Grep Tool

- [x] Add `output_mode` parameter (content, files_with_matches, count) with files_with_matches as default
- [x] Add `head_limit` parameter to cap output lines (default: 250)
- [x] Add `-A`/`-B`/`-C`/`context` parameters for context lines
- [x] Add `type` parameter for file type filtering (rg --type)
- [x] Add `multiline` parameter for cross-line matching (rg -U)
- [x] Add `offset` parameter to skip first N results
- [x] Cache rg availability check instead of running `rg --version` every call
- [ ] Improve description with usage guidance
