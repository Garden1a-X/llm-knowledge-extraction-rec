# Phase 3 V2 Validation - 5% Test Analysis Report

**Date**: 2026-01-11
**Test Dataset**: 5% (170 movies)
**Vocabulary**: v2 (180 entities)
**Model**: gpt-4o-mini

---

## Executive Summary

**KEY FINDING**: Vocabulary v2 is **highly effective** (83.35% coverage after correction), but **gpt-4o-mini has significant LLM review accuracy issues**.

### Results at a Glance

| Metric | Original | Corrected | Improvement |
|--------|----------|-----------|-------------|
| **Coverage** | **73.09%** | **83.35%** | **+10.26%** |
| Valid Entities | 1168 (73.1%) | 1332 (83.4%) | +164 |
| NEW_ Entities | 318 (19.9%) | 176 (11.0%) | -142 |
| Invalid Entities | 112 (7.0%) | 90 (5.6%) | -22 |
| **Total KPs** | 1598 | 1598 | - |

**Conclusion**: v2 vocabulary would achieve **~83% coverage** with proper LLM implementation.

---

## Problem Breakdown

### Issue 1: False NEW_ Entities (44.7% of all NEW_)

**Root Cause**: LLM failed to properly check vocabulary during self-review.

**Evidence**:
- 142 out of 318 NEW_ entities (44.7%) actually exist in vocabulary v2
- Example: LLM output `NEW_dark` even though "dark" is clearly listed in mood vocabulary

**Top False NEW_ Corrections**:
```
Relation              False NEW_    Should Be
-----------------------------------------------
mood                  75 entities   adventurous, dark, humorous, dramatic, whimsical, etc.
visual_theme          23 entities   romance, adventure, drama, thriller, mystery, etc.
design_element        19 entities   geometric shapes, modern casual, period costume, etc.
depicted_subject       6 entities   romantic outdoors, fantasy creatures, etc.
```

**Impact**:
- These 142 false NEW_ caused coverage to appear 8.89% lower than reality
- True coverage: 81.98% (before invalid format fixes)

### Issue 2: Invalid Format Mismatches (19.6% of invalids)

**Root Cause**: LLM used wrong format (space vs underscore).

**Evidence**:
- 22 out of 112 invalid entities were just format mismatches
- Examples:
  - `"romantic outdoors"` → should be `"romantic_outdoors"`
  - `"dramatic lighting"` → should be `"dramatic_lighting"`
  - `"vibrant contrasts"` → should be `"vibrant_contrasts"`

**Impact**:
- These format issues caused 1.38% additional coverage loss

### Combined Impact

**Original**: 73.09% coverage
**After false NEW_ fix**: 81.98% (+8.89%)
**After format fix**: 83.35% (+1.37%)
**Total improvement**: +10.26%

---

## Corrected Results Analysis

### Coverage by Category (After Correction)

| Category | Count | Percentage | Notes |
|----------|-------|------------|-------|
| **Valid (in vocab)** | 1332 | **83.35%** | ✅ Target: 85-90% |
| NEW_ (true new) | 176 | 11.01% | Legitimate new entities |
| Invalid (errors) | 90 | 5.63% | LLM mistakes |

### Remaining NEW_ Entities (176 total)

**High Frequency (≥3 occurrences)**:
- `character_type:NEW_action_characters` (3x) - *Note: "action characters" exists in action_behaviors*
- `additional_elements:NEW_vibrant_contrasts` (3x) - *Note: "vibrant_contrasts" exists in color_palette*
- `mood:NEW_intense` (2x)
- `mood:NEW_suspenseful` (2x) - *Similar to existing "suspense"*
- `lighting:NEW_soft_lighting` (2x)

**Analysis**:
- Many are **relation mismatches** (entity exists but in wrong relation)
- Some are **semantic duplicates** (e.g., suspenseful ≈ suspense)
- ~50% could potentially be eliminated with better prompting

**True New Entities** (estimated ~88):
- `visual_theme:NEW_horror`, `NEW_comedy`, `NEW_musical` (genres not in visual_theme)
- `mood:NEW_intense`, `NEW_tense`
- `texture:NEW_smooth_texture`

### Remaining Invalid Entities (90 total)

**Top Invalid Patterns**:

1. **Wrong Relations** (47 entities):
   - `action_characters:action characters` → should be `character_type`
   - `action_poses:action poses` → should be `action_behaviors`
   - `environment_interactions:environment interactions` → should be `action_behaviors`
   - `character:character` → invalid relation entirely

2. **Still Format Issues** (23 entities):
   - `visual_theme:horror` → should be in genre
   - `visual_theme:comedy` → should be in genre
   - `visual_theme:sci-fi` → should be in genre

3. **Other Errors** (20 entities):
   - `emotional_moments:the couple appears intimate` → descriptive sentence, not entity
   - `lighting:bright_lighting` vs `bright lighting` → inconsistent formatting

---

## Vocabulary v2 Effectiveness

### Performance by Relation

| Relation | v1 Entities | v2 Entities | Coverage Impact |
|----------|-------------|-------------|-----------------|
| mood | 12 | 17 (+5) | ✅ High impact (75 false NEW_ corrected) |
| visual_theme | 9 | 13 (+4) | ✅ High impact (23 false NEW_ corrected) |
| design_element | 8 | 10 (+2) | ✅ Good impact (19 false NEW_ corrected) |
| genre | 7 | 9 (+2) | ✅ Good coverage |
| lighting | 7 | 8 (+1) | ✅ Helpful (2 false NEW_ corrected) |
| color_palette | 12 | 13 (+1) | ✅ Complete coverage |

**Verdict**: v2 expansion was **highly effective**. The 15 added entities covered 142 false NEW_ instances.

---

## Root Cause Analysis

### Why Did LLM Make So Many Mistakes?

**1. gpt-4o-mini Limitation** (Primary Cause):
- Model is **too weak** for reliable vocabulary lookup in long lists
- Evidence: Said "dark not in mood list" when it's clearly listed
- Impact: 44.7% false NEW_ rate

**2. Vocabulary Presentation Format**:
```
【Relation 11: mood】
Standard Entities (17 total):
- action
- adventurous
- dark          ← Clearly visible but LLM missed it!
- dark humor
- dramatic
...
```

- Format is clear to humans, but LLM struggles with 180-entity lookup
- Possible: **Attention mechanism limitation** in small model

**3. Format Inconsistency**:
- Vocabulary mixes underscores and spaces: `"dark_backgrounds"` vs `"bright lighting"`
- LLM sometimes outputs `"romantic outdoors"` when vocab has `"romantic_outdoors"`
- No clear standard enforced

---

## Recommendations

### Immediate Actions (For Phase 3 20% Validation)

**Option A: Use gpt-4 (RECOMMENDED)**
- Cost: ~$10-15 for 20% dataset
- Expected: <10% false NEW_ rate (vs 44.7% with 4o-mini)
- **ROI**: Worth the cost for accuracy

**Option B: Use 4o-mini + Post-processing**
- Cost: ~$0.50 for 20% dataset
- Apply correction script automatically
- Pros: Cheap and fast
- Cons: "Dirty" solution, doesn't fix root cause

**My Recommendation**: Use **gpt-4** for 20% validation to verify v2 effectiveness without post-processing bias.

### Medium-term Fixes (For Phase 4)

1. **Improve Prompt Format**:
   - Add instruction: "CRITICAL: Before adding NEW_ prefix, carefully check the EXACT entity name in the vocabulary list above!"
   - Use structured format: "If entity X appears in the list, use it EXACTLY as shown (including spaces/underscores)"

2. **Standardize Format**:
   - Decision: All entities use underscores OR all use spaces
   - Update vocabulary to be consistent
   - Add validation in post-processing

3. **Use gpt-4 for Final Extraction**:
   - Phase 4 (80% dataset) should use gpt-4
   - Cost: ~$50-70 for full dataset
   - Worth it for research quality

### Long-term (Post-KDD)

1. **Fine-tune or Few-shot Examples**:
   - Add examples showing how to properly check vocabulary
   - Few-shot: "Example: If you see 'dark' in the image, check mood list → 'dark' exists → use 'dark' not 'NEW_dark'"

2. **Two-stage Process**:
   - Stage 1: Fast extraction with 4o-mini
   - Stage 2: Quality check with gpt-4 on NEW_ entities only
   - Hybrid: Cost-effective + accurate

---

## Projected 20% Validation Results

### Scenario 1: With gpt-4
- Expected coverage: **83-85%**
- NEW_ rate: **10-12%**
- Invalid rate: **3-5%**
- Decision: ✅ **Proceed to Phase 4** if ≥83%

### Scenario 2: With 4o-mini + Post-processing
- Expected coverage: **81-83%** (after correction)
- NEW_ rate: **11-13%** (after correction)
- Invalid rate: **5-7%** (after correction)
- Decision: ⚠️ Borderline, may need v3 or prompt improvements

---

## Next Steps

### Step 1: 20% Validation (Choose One)

**Plan A (Recommended)**: Use gpt-4
```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --model gpt-4o \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --percentage 20.0 \
  --workers 10 \
  --output results/phase3_20percent_v2_gpt4.json
```
- Time: ~10-15 min
- Cost: ~$10-15
- Expected coverage: 83-85%

**Plan B (Budget)**: Use 4o-mini + correction
```bash
# Run extraction
python scripts/phase3_validate_vocabulary_concurrent.py \
  --model gpt-4o-mini \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --percentage 20.0 \
  --workers 15 \
  --output results/phase3_20percent_v2_mini.json

# Apply correction
python scripts/fix_false_new_entities.py \
  --input results/phase3_20percent_v2_mini.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --output results/phase3_20percent_v2_mini_corrected.json
```
- Time: ~5-7 min
- Cost: ~$0.50
- Expected coverage: 81-83% (after correction)

### Step 2: Decide Based on Results

- **If coverage ≥ 85%**: ✅ Proceed to Phase 4 (80% extraction with gpt-4)
- **If coverage 80-84%**: ⚠️ Acceptable but consider minor v2.1 expansion
- **If coverage < 80%**: ❌ Need v3 or prompt improvement

### Step 3: Phase 4 Full Extraction

```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --model gpt-4o \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --percentage 80.0 \
  --workers 10 \
  --save-interval 20 \
  --output results/phase4_80percent_extraction.json
```
- Time: ~25-35 min (vs 5-6 hours single-threaded)
- Cost: ~$50-70
- Expected: 3400+ movies with 83-85% coverage

---

## Files Generated

1. ✅ `results/phase3_5percent_test_v2.json` - Original results
2. ✅ `results/phase3_5percent_test_v2_corrected.json` - Corrected results
3. ✅ `results/false_new_correction_report.json` - Detailed correction log
4. ✅ `results/phase3_v2_5percent_analysis_report.md` - This report
5. ✅ `scripts/fix_false_new_entities.py` - Correction script (reusable)

---

## Conclusion

**Vocabulary v2 is effective** - achieves **83.35% coverage** on 5% test data after correction.

**Main Issue**: gpt-4o-mini is **too weak** for reliable vocabulary checking, producing 44.7% false NEW_ rate.

**Recommendation**: Use **gpt-4** for 20% validation ($10-15) to get clean results without post-processing bias.

**Path Forward**:
1. Run 20% with gpt-4 → Verify 83-85% coverage → Proceed to Phase 4 (80% with gpt-4)
2. Total cost: ~$60-85 for complete high-quality extraction
3. Timeline: Can complete within 1 hour with multi-threading

**For KDD deadline (2/9)**, this is feasible and will deliver high-quality knowledge graph data.

---

**Generated**: 2026-01-11
**Author**: Claude (Automated Analysis)
