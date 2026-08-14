## Summary

Describe the change and the problem it addresses.

## Scope

- Affected modules:
- Data sources or date ranges:
- Public metrics or figures changed: yes / no

## Validation

List the exact commands run and summarize the results.

```bash
pytest -q
```

## Research integrity checklist

- [ ] Every feature uses only information available before its delivery day's 11:00 EPT gate.
- [ ] Feature availability is validated with timestamps where applicable.
- [ ] Raw DART labels and tails remain unmodified.
- [ ] Train, validation, and test boundaries remain chronological.
- [ ] The test period did not influence feature, model, label, or threshold selection.
- [ ] Changes affecting results regenerate and explain all public metrics and figures.
- [ ] Transaction costs, execution assumptions, missing data, and exclusions are documented.

Mark an item not applicable and explain why when necessary.

## Data and security

- [ ] No API keys, credentials, private paths, or raw unapproved datasets are included.
- [ ] New dependencies are necessary and pinned appropriately.
- [ ] Tests avoid network access unless explicitly marked as integration tests.

## Documentation

Describe any README, methodology, or reproduction instructions that were updated.
