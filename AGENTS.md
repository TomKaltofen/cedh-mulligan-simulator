# AGENTS.md

Must read [README.md](README.md) first.

This project uses the mloda framework. Assume any given task is related to mloda.

## Environment

```bash
source .venv/bin/activate
```

## Dependencies

Use `uv` to install dependencies:
```bash
uv sync --all-extras
```

## Running checks

Use `tox` to run all checks:
```bash
tox
```

### Run individual checks

```bash
pytest
ruff format --check --line-length 120 .
ruff check .
mypy --strict --ignore-missing-imports .
bandit -c pyproject.toml -r -q .
```

## Claude Code Skills

The mloda-registry provides Claude Code skills that assist with plugin development:

- https://github.com/mloda-ai/mloda-registry/tree/main/.claude/skills/

When helping with FeatureGroups, ComputeFrameworks, or Extenders, leverage these skills for pattern guidance and best practices.

Consider generating project-specific skills for your own plugin repository to provide tailored AI assistance for your implementation patterns and conventions.

## mloda Feature Group Guides

Before building or adapting feature groups, consult the guides at:
https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides/

### Decision Tree — Choosing a Pattern

Use [09-create-feature-group.md](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/09-create-feature-group.md) to pick the right pattern:

1. **Does it load/generate data with no dependencies?** → Root Feature ([01-root-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/01-root-features.md))
   - Use `DataCreator({"feat_a", "feat_b"})` from `input_data()` for synthetic/generated data.
   - Keys in the `DataCreator` set **must** match keys in the dict returned by `calculate_feature`.
2. **Does it transform one or more existing features?** → Derived Feature ([02-derived-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/02-derived-features.md))
   - Return dependencies from `input_features()`, compute in `calculate_feature()`.
   - Default matching is by **class name** (e.g. class `DoubledValue` matches feature `"DoubledValue"`).
3. **Should it be reusable via naming pattern (`input__operation`)?** → Chained Feature ([03-chained-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/03-chained-features.md))
4. **Does it produce multiple output columns?** → Multi-output (`~` separator) ([05-multi-output-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/05-multi-output-features.md))

### Core Concepts

- **Options** ([11-options](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/11-options.md)): `group` values are hashed and affect feature resolution. `context` values are metadata only. Nested dicts (like card registries) must go in `context`, not `group`.
- **calculate_feature()** ([12-calculate-feature](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/12-calculate-feature.md)): Receives `data` (DataFrame/dict with dependencies computed) and `features` (FeatureSet with requested features, options, filters).
- **Feature Naming** ([13-feature-naming](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/13-feature-naming.md)): Class name (default), `feature_names_supported()` (explicit set), or `PREFIX_PATTERN` (regex). Separators: `__` (chain), `~` (multi-output), `&` (multi-input).
- **Feature Matching** ([14-feature-matching](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/14-feature-matching.md)): Priority order — input data match → exact class name → prefix match → explicit names. Exactly one FeatureGroup must match per feature name.

### Testing Strategy

Follow the 3-level approach from [10-testing-guide](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/10-testing-guide.md):

1. **Unit** (fast): Test `match_feature_group_criteria()`, `input_features()`, config methods.
2. **Framework** (medium): Test `calculate_feature()` with real DataFrames.
3. **Integration** (slow): Test full pipeline via `mloda.run_all()`.

### All Pattern Guides

| Pattern | Guide |
|---------|-------|
| Root features (data sources) | [01-root-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/01-root-features.md) |
| Simple derived features | [02-derived-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/02-derived-features.md) |
| Chained features (`input__op`) | [03-chained-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/03-chained-features.md) |
| Multi-input features (`a&b__op`) | [04-multi-input-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/04-multi-input-features.md) |
| Multi-output features (`feat~N`) | [05-multi-output-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/05-multi-output-features.md) |
| Artifact features (fitted state) | [06-artifact-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/06-artifact-features.md) |
| Index features (time/group-by) | [07-index-features](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/07-index-features.md) |
| Links/Joins | [08-links-joins](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/08-links-joins.md) |
| Framework-specific | [09-framework-specific](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/09-framework-specific.md) |
| Testing guide | [10-testing-guide](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/10-testing-guide.md) |
| Options (group vs context) | [11-options](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/11-options.md) |
| calculate_feature() | [12-calculate-feature](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/12-calculate-feature.md) |
| Feature naming | [13-feature-naming](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/13-feature-naming.md) |
| Feature matching | [14-feature-matching](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/14-feature-matching.md) |
| Filter concepts | [15-filter-concepts](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/15-filter-concepts.md) |
| Validators | [16-validators](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/16-validators.md) |
| Data connection matching | [17-data-connection-matching](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/17-data-connection-matching.md) |
| Data types | [18-datatypes](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/18-datatypes.md) |
| Domain disambiguation | [19-domain](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/19-domain.md) |
| Versioning | [20-versioning](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/20-versioning.md) |
| Experimental shortcuts | [21-experimental-shortcuts](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/21-experimental-shortcuts.md) |
| Feature config (JSON/AI) | [22-feature-config](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/22-feature-config.md) |

### Plugin Journey Guides

| Guide | When to Use |
|-------|-------------|
| [01-use-existing-plugin](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/01-use-existing-plugin.md) | Using plugins from the registry |
| [02-discover-plugins](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/02-discover-plugins.md) | Finding available plugins |
| [03-create-plugin-in-project](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/03-create-plugin-in-project.md) | Adding feature groups inline (no separate package) |
| [04-create-plugin-package](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/04-create-plugin-package.md) | Packaging plugins for distribution |
| [05-share-with-team](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/05-share-with-team.md) | Sharing via private git repo |
| [06-publish-to-community](https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/06-publish-to-community.md) | Publishing to community registry |
