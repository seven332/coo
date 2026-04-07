# TODO

## Grep Tool

- [x] Add `output_mode` parameter (content, files_with_matches, count) with files_with_matches as default
- [x] Add `head_limit` parameter to cap output lines (default: 250)
- [x] Add `-A`/`-B`/`-C`/`context` parameters for context lines
- [ ] Add `type` parameter for file type filtering (rg --type)
- [ ] Add `multiline` parameter for cross-line matching (rg -U)
- [ ] Add `offset` parameter to skip first N results
- [ ] Cache rg availability check instead of running `rg --version` every call
- [ ] Improve description with usage guidance
