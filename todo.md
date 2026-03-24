# cEDH Mulligan Simulator — Prioritized Backlog

## Tier 1 — Code Health (quick wins)

- [ ] **Remove debug print** — `cedh_mulligan_simulator/feature_groups/mulligan/hand_generator.py:44`
  - Replace `print("Create hands:", len(hands))` with `logging.debug(...)` or remove
- [ ] **Update stale docs** — `README.md` and `docs/getting-started.md` still contain `mloda-plugin-template` boilerplate
  - Replace with a project-specific description of the cEDH mulligan simulator

---

## Tier 2 — Simulation Accuracy

- [ ] **Lake of the Dead — implement `sacrifice_swamp` requirement**
  - File: `cedh_mulligan_simulator/mana.py` → `_compute_battlefield_mana()`
  - `LAKE_OF_THE_DEAD` has `requires="sacrifice_swamp"` but `_compute_battlefield_mana` ignores it
  - Currently silently gives 4B on T2+ regardless of whether a swamp is available
  - Fix: check if a black-mana land exists on battlefield before granting 4B; mark that land sacrificed

- [ ] **Variable mana — Rain of Filth / Songs of the Damned**
  - File: `card_database/black/__init__.py`
  - Both produce `Mana()` (zero) because the actual amount depends on runtime board state
  - Add explicit comments explaining they are excluded from mana calculation intentionally

- [ ] **Urza's Saga — use the sac-search mechanic**
  - File: `card_database/lands/__init__.py`
  - `URZAS_SAGA` is simplified (`t1_mana=Mana(1), t2_mana=Mana(1)`) but `mana.py` already has the
    `sac_search_turn`/`sac_search_artifact_cmcs` system used by `URZAS_CAVE`
  - Add `sac_search_turn=3, sac_search_artifact_cmcs=(0, 1)` to model chapter 3 tutor

---

## Tier 3 — Test Coverage Gaps

- [ ] **Implement `test_land_gemstone` stub**
  - File: `tests/test_integration_mloda_specific_hands.py`
  - Currently just `pass` — add non-trivial assertions for:
    - Gemstone Caverns: exile-from-hand enables T1 casting
    - Lake of the Dead: T3 produces 4B (not 5B) when sacrificing a swamp; 0 when no swamp

- [ ] **Add Turn3 integration test**
  - File: `tests/test_turn_evaluation.py`
  - Add `test_turn3_feature()` mirroring the existing T1/T2 tests

- [ ] **Test `draw_per_turn=True`**
  - File: `tests/test_integration_mloda_specific_hands.py`
  - All existing `TestBraidsSpecificHands` tests use `draw_per_turn=False`
  - Add at least one T2+ test with `draw_per_turn=True` to cover the draw logic path

- [ ] **Test `MulliganResult` with `max_keep_turn=3`** — only `max_turn=2` default is tested
- [ ] **Test `HandGenerator` deck_size override** — no test uses a non-99 deck size

---

## Tier 4 — Complete A/B Scenarios

- [ ] **Wire up all 13 scenarios in `run_simulation.py`**
  - Currently only `baseline` and `no_lions_eye_diamond` are in `SCENARIOS`
  - Add: `no_jeweled_lotus`, `no_gemstone_caverns`, `no_jeweled_amulet`, `no_peat_bog`,
    `no_0mana_creatures`, `1cmc_creatures`, `no_grief`, `no_deepwood_legate`, `no_offalsnout`,
    `peat_bog_to_swamp`, `no_drum`, `no_mantle`

- [ ] **Refactor `run_simulation.py` to remove duplicate helpers** (lower priority)
  - `wilson_interval()` duplicates `confidence_interval.py`
  - `create_scenario_comparison_plot()` duplicates `ScenarioComparisonPlot` FG
  - `create_card_delta_table()` duplicates `CardDeltaTable` FG

---

## Tier 5 — SacrificeFodder Feature Group

- [ ] **Extract `t1_has_fodder` into a `SacrificeFodder` feature group**
  - Currently computed as a side effect inside `TurnFeatureBase`
  - Extract as a proper derived FG that depends on `Turn1` output
  - Enables "T1 Fodder %" column in output tables
  - Add a pass-through FG class so `t1_has_fodder` can be requested independently

---

## Tier 6 — Future / Longer-term

- [ ] **Polars compute framework** — swap `PandasDataFrame` for `PolarsDataFrame` at 1M+ scale
- [ ] **OpenTelemetry lineage extender** — trace which cards contributed to a keep decision
- [ ] **Multi-color commander support** — populate `card_database/blue/`, `green/`, etc.
- [ ] **Community packaging** — publish mechanics feature groups to mloda-registry
