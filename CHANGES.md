# Reorganization Changes

## What Changed (January 21, 2026)

### 1. Folder Renamed
**Before**: `AGI-ICML-2026-ESR-paper`
**After**: `AGI-ICML-2026-ESR-paper-code`

**Reason**: Clarifies this contains code (not paper LaTeX, data, etc.)

### 2. Experiments Renumbered

The "appendices" folder has been broken out into numbered experiments 6-8:

| Old Name | New Name | Paper Section |
|----------|----------|---------------|
| `appendices/self-correction-activation-statistics/` | `experiment_6_sequential_activations/` | §3.6 + Appendix A.4 |
| `appendices/otd-activation-statistics/` | `experiment_7_otd_statistics/` | Appendix A.3.5 |
| `random_ablation_control/` | `experiment_8_random_ablation_control/` | Appendix A.3.6 |

### 3. Removed Nested Folder

**Issue**: There was a nested `experiments/AGI-ICML-2026-ESR-paper/` folder inside the main folder (likely from a copy operation).

**Fixed**: Removed the nested folder to clean up structure.

### 4. New Documentation Added

- `README_EXPERIMENTS.md`: Quick reference for all experiments
- `STRUCTURE.md`: Explains folder organization and naming
- `CHANGES.md`: This file

## Benefits of New Structure

### ✅ Clear Numbering
All experiments now numbered 1-8, matching their order in the paper

### ✅ Easier Navigation
```bash
# Old way (confusing)
cd appendices/self-correction-activation-statistics

# New way (clear)
cd experiment_6_sequential_activations
```

### ✅ Better Mapping to Paper
| Paper Section | Folder |
|---------------|--------|
| §3.1 | `experiment_1_esr.py` |
| §3.2 | `experiment_2_multi_boost.py` |
| §3.3 | `experiment_5_prompt_variants.py` |
| §3.4 | `experiment_3_off_topic_detectors/` |
| §3.5 | `experiment_4_finetuning/` |
| §3.6 | `experiment_6_sequential_activations/` |
| Appendix A.3.5 | `experiment_7_otd_statistics/` |
| Appendix A.3.6 | `experiment_8_random_ablation_control/` |

### ✅ Consistent Naming
All experiments follow pattern: `experiment_N_descriptive_name`

## Updated Documentation

All documentation has been updated to reflect new structure:
- ✅ `README.md`: Updated directory structure
- ✅ `MANIFEST.md`: Updated experiment paths
- ✅ `PAPER_MAPPING.md`: Updated folder references
- ✅ `QUICKSTART.md`: Updated navigation examples
- ✅ `scripts/run_all_experiments.sh`: Updated experiment paths

## What Stayed the Same

- All core infrastructure files (unchanged)
- All experiment scripts (unchanged)
- All data files (unchanged)
- All plotting scripts (unchanged)
- Main experiments 1-5 naming (unchanged)

## Migration Guide

If you had old paths in scripts or notes, here's the mapping:

```bash
# Old → New
appendices/self-correction-activation-statistics → experiment_6_sequential_activations
appendices/otd-activation-statistics → experiment_7_otd_statistics
random_ablation_control → experiment_8_random_ablation_control
AGI-ICML-2026-ESR-paper → AGI-ICML-2026-ESR-paper-code
```

## Verification

To verify the structure is correct:

```bash
# Should show experiments 1-8
ls -d experiment_* | sort

# Should show all documentation
ls *.md

# Should show correct folder name
basename "$(pwd)"
# Output: AGI-ICML-2026-ESR-paper-code
```

## Summary

- **Folder renamed**: Added "-code" suffix for clarity
- **Experiments renumbered**: 6-8 broken out from appendices
- **Nested folder removed**: Cleaned up accidental duplication
- **Documentation updated**: All references corrected
- **Consistency improved**: Clear 1-8 numbering throughout

All code remains functional - only organization changed!
