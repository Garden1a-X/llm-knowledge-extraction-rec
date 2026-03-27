# IMPORTANT: DO NOT USE MMRec

**Decision Date:** 2026-01-21

## Summary

**We will NOT use the MMRec framework for any baselines in this project.**

## Reasons

1. **Data format incompatibility**: MMRec requires specific data formats that don't align with our RecBole-based pipeline
2. **Evaluation protocol issues**: MMRec's evaluation mode (full ranking vs. uni100) is difficult to configure and verify
3. **Code complexity**: MMRec's internal data structures are opaque and hard to debug
4. **Maintenance burden**: Adding MMRec support requires significant patching and custom code
5. **Consistency**: We need all baselines to use the same evaluation protocol for fair comparison

## Alternative Approaches

### For VBPR Baseline
- **DO NOT** implement VBPR using MMRec
- Options:
  1. Use RecBole's built-in VBPR implementation (if available)
  2. Implement VBPR from scratch using our existing framework
  3. Use published results from the original VBPR paper

### For MMGCN and Other Visual Baselines
- Implement using RecBole framework
- Use our standardized data pipeline (RecBole format)
- Use our standardized evaluation protocol (same as LightGCN, KGAT, MKGAT)

## Deleted Files/Commits

The following MMRec-related files and commits have been removed:
- `baselines/prepare_mmrec_data.py`
- `baselines/run_vbpr.py`
- `baselines/run_vbpr_5trials.py`
- `baselines/run_vbpr_5trials.sh`
- `baselines/run_vbpr_mmrec.sh`
- `baselines/setup_and_run_vbpr.sh`
- `baselines/vbpr_config.yaml`
- `baselines/aggregate_vbpr_results.py`
- `baselines/convert_recbole_to_mmgcn.py`
- `baselines/VBPR_README.md`

Git commits reverted: from 125a11a to 4cd1564 (10 commits)

## Moving Forward

- Focus on baselines that can be implemented using RecBole
- Maintain consistency in data format and evaluation protocol
- Document any exceptions clearly

---

**If you see this file, DO NOT attempt to re-add MMRec support without explicit approval.**
