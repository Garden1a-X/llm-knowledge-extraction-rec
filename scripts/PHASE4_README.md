# Phase 4: Full Production Extraction

Final extraction for movie recommendation knowledge graph using validated vocabulary v2.

## Overview

Phase 4 uses a **simplified extraction approach**:
- ✅ **No NEW_ mechanism** - just extract using vocabulary as reference
- ✅ **No self-review** - faster and cheaper
- ✅ **Post-processing filter** - only keep vocabulary entities
- ✅ **Multi-threaded** - 15-20x faster than sequential

**Expected Results**:
- Coverage: **85-90%** (based on 20% validation)
- Clean vocabulary-only entities
- Ready for RecBole recommendation training

---

## Quick Start

### Full 100% Extraction

```bash
# Step 1: Extract all movies (35-45 minutes, $2.5-3.5)
python scripts/phase4_full_extraction.py \
  --poster_dir /data/xuao/ml-1m/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --model gpt-4o-mini \
  --percentage 100.0 \
  --workers 15 \
  --save-interval 20 \
  --output results/phase4_full_extraction.json

# Step 2: Filter to vocabulary-only entities (<1 minute)
python scripts/filter_to_vocabulary.py \
  --input results/phase4_full_extraction.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --output results/phase4_full_extraction_filtered.json \
  --report results/phase4_filtering_report.json
```

**Total time**: ~40-50 minutes
**Total cost**: ~$2.5-3.5
**Final dataset**: ~3400 movies, ~28,000-30,000 knowledge points, 85-90% retention

---

## Step-by-Step Guide

### Step 1: Full Extraction

**What it does**:
- Uses simplified Phase 4 prompt (no NEW_, no self-review)
- Extracts 8-12 knowledge points per movie
- LLM uses vocabulary as reference (best-effort matching)
- Saves raw results (may include entities not in vocabulary)

**Parameters**:
- `--percentage 100.0`: Extract all movies (~3400)
- `--workers 15`: Concurrent threads (adjust based on API limits)
- `--save-interval 20`: Checkpoint frequency
- `--model gpt-4o-mini`: Recommended for cost/performance balance

**Expected output**:
- Raw extraction with ~85-90% vocabulary match
- ~10-15% entities may not match vocabulary (typos, format issues)

---

### Step 2: Vocabulary Filtering

**What it does**:
- Keeps only entities that exist in vocabulary v2
- Handles format variants (space/underscore, plural/singular)
- Removes entities not in vocabulary
- Generates filtering statistics

**Expected filtering**:
- Retention rate: **85-90%** (based on 20% validation)
- Format corrections: ~5-10%
- Removed non-vocab: ~10-15%

**Output statistics**:
```
Total KPs: ~32,000
Kept (in vocabulary): ~28,000-30,000 (85-90%)
Removed (not in vocab): ~3,000-5,000 (10-15%)
Format corrections: ~1,500-2,000
```

---

## Prompt Differences

### Phase 3 (Validation)
```
- Complex prompt with NEW_ mechanism
- Self-review (3 steps: Draft → Review → Final)
- Longer output (~1000 tokens)
- Purpose: Analyze coverage, find gaps
```

### Phase 4 (Production)
```
- Simple prompt: "Extract using vocabulary"
- No self-review, direct output (~500 tokens)
- Faster, cheaper, cleaner
- Purpose: Final production data
```

**Why simpler is better**:
1. **Cost**: ~50% cheaper (fewer tokens)
2. **Speed**: ~30% faster (no CoT reasoning)
3. **Quality**: LLM focuses on matching, not inventing
4. **Post-processing**: Filtering guarantees vocabulary-only

---

## Output Format

### Raw Extraction (`phase4_full_extraction.json`)

```json
{
  "phase": "phase4_full_extraction",
  "config": {...},
  "results": [
    {
      "recbole_id": 1,
      "original_movie_id": 1,
      "status": "success",
      "num_knowledge_points": 10,
      "knowledge_points": [
        {"relation": "mood", "entity": "romantic"},
        {"relation": "color_palette", "entity": "warm_colors"},
        {"relation": "visual_theme", "entity": "romance"},
        ...
      ],
      "raw_output": "...",
      "timestamp": "2026-01-11T..."
    },
    ...
  ],
  "total_movies": 3413,
  "successful": 3400,
  "failed": 13
}
```

### Filtered Output (`phase4_full_extraction_filtered.json`)

```json
{
  "phase": "phase4_full_extraction",
  "config": {...},
  "filtered": true,
  "filter_stats": {
    "total_knowledge_points": 32000,
    "kept_knowledge_points": 28500,
    "removed_knowledge_points": 3500,
    "retention_rate": 0.8906,
    "retention_percentage": 89.06
  },
  "results": [
    {
      "recbole_id": 1,
      "num_knowledge_points": 9,  // After filtering
      "knowledge_points": [
        {"relation": "mood", "entity": "romantic"},
        {"relation": "color_palette", "entity": "warm_colors"},
        ...
      ]
    },
    ...
  ],
  "total_movies": 3413,
  "total_knowledge_points": 28500
}
```

---

## Resume Interrupted Extraction

If extraction is interrupted, just re-run the same command:

```bash
python scripts/phase4_full_extraction.py \
  --poster_dir /data/xuao/ml-1m/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --percentage 100.0 \
  --workers 15 \
  --output results/phase4_full_extraction.json
```

**It will automatically**:
- Load existing results
- Skip successfully processed IDs
- Continue from where it left off
- Merge with previous results

---

## Performance Tuning

### Workers

- **Too low (5-10)**: Slow, underutilized
- **Optimal (15-20)**: Fast, good throughput
- **Too high (25+)**: May hit API rate limits

**Test**: Start with 15, increase to 20 if no errors

### Save Interval

- **Lower (10-15)**: More frequent saves, safer
- **Higher (20-30)**: Less I/O overhead, faster

**Recommended**: 20 for 100% extraction

### Model Choice

| Model | Cost | Speed | Quality | Recommendation |
|-------|------|-------|---------|----------------|
| gpt-4o-mini | $ | Fast | Good | ✅ **Recommended** |
| gpt-4o | $$$ | Fast | Best | Optional (overkill) |
| gpt-4-turbo | $$$$ | Medium | Best | Not needed |

**For Phase 4**: gpt-4o-mini is sufficient (post-filtering handles errors)

---

## Expected Costs

| Dataset | Model | Time | Cost |
|---------|-------|------|------|
| 100% (~3400) | gpt-4o-mini | 35-45 min | **$2.5-3.5** |
| 100% (~3400) | gpt-4o | 35-45 min | $50-70 |

**Recommendation**: Use gpt-4o-mini + filtering (10-20x cheaper, same final quality)

---

## Troubleshooting

### Rate Limit Errors
```bash
# Reduce workers
--workers 10  # or even 5
```

### Low Retention Rate (<80%)
- Check vocabulary file path
- Verify vocabulary v2 is being used
- Review filtering report for patterns

### Memory Issues
```bash
# Reduce workers and save more frequently
--workers 10 --save-interval 10
```

### Extraction Hangs
- Check API connectivity
- Verify API key and base URL
- Monitor for error messages in output

---

## Next Steps After Extraction

1. **Analyze filtered results**:
```bash
python scripts/analyze_extraction_stats.py \
  --input results/phase4_full_extraction_filtered.json \
  --output results/phase4_stats_report.json
```

2. **Convert to RecBole format**:
```bash
python scripts/convert_to_recbole.py \
  --input results/phase4_full_extraction_filtered.json \
  --output data/recbole/ml-1m/ml-1m.kg
```

3. **Train recommendation model**:
```bash
python run_recbole.py --config configs/kgat_ml1m.yaml
```

---

## Files

1. ✅ `scripts/phase4_full_extraction.py` - Main extraction script
2. ✅ `scripts/filter_to_vocabulary.py` - Filtering script
3. ✅ `src/extraction/prompts.py` - Updated with Phase 4 prompts
4. ✅ `results/standard_entity_vocabulary_v2.json` - Vocabulary (180 entities)

---

## Summary

**Phase 4 Strategy**:
1. Simple extraction using vocabulary as reference
2. Post-processing filtering to vocabulary-only
3. Fast, cheap, high-quality results

**Expected Final Dataset**:
- 3400 movies
- 28,000-30,000 knowledge points
- 85-90% vocabulary coverage
- Ready for RecBole training

**Total Time**: ~45 minutes
**Total Cost**: ~$3
**Quality**: Production-ready for KDD paper

---

**Ready to extract? Run the commands above!** 🚀
