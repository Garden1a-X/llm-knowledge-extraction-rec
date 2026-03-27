# Multi-threaded Phase 3 Validation

## Overview

`phase3_validate_vocabulary_concurrent.py` is a multi-threaded version of the Phase 3 validation script that uses concurrent API calls to speed up extraction.

**Speed improvement**: ~10-20x faster than single-threaded version

## Key Features

- **Multi-threaded API calls**: Concurrent processing with configurable workers
- **Thread-safe**: Lock-protected result collection and file saving
- **Incremental saving**: Periodic checkpoints every N completions
- **Error recovery**: Skip already processed IDs, retry errors optionally
- **Progress tracking**: Real-time progress bar with coverage stats

## Usage

### Basic Usage (with vocabulary v2)

```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --output results/phase3_20percent_validation_v2.json \
  --workers 15
```

### Key Parameters

- `--workers`: Number of concurrent threads (default: 10, recommended: 10-20)
  - Too low: Slow extraction
  - Too high: May hit API rate limits
  - Recommended: Start with 15, adjust based on API performance

- `--save-interval`: Save checkpoint every N completions (default: 10)
  - Lower value: More frequent saves, safer but slower I/O
  - Higher value: Less I/O overhead, but lose more progress on crash
  - Recommended: 10-20

- `--percentage`: Percentage of movies to sample (default: 20.0)
  - For Phase 3 validation: 20.0 (~683 movies)
  - For Phase 4 full extraction: 80.0 (~3400 movies)

- `--retry-errors`: Retry previously failed extractions
  - Default: Skip all processed IDs (including errors)
  - With flag: Retry error IDs, skip only successful ones

### Examples

**Quick test with 5% and 20 workers:**
```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --percentage 5.0 \
  --workers 20 \
  --output results/phase3_5percent_test_v2.json
```

**Full Phase 3 validation (20%) with v2:**
```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --percentage 20.0 \
  --workers 15 \
  --output results/phase3_20percent_validation_v2.json
```

**Phase 4 full extraction (80%):**
```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --percentage 80.0 \
  --workers 15 \
  --save-interval 20 \
  --output results/phase4_80percent_extraction.json
```

**Resume interrupted extraction:**
```bash
# Same command as before - will automatically skip processed IDs
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --output results/phase3_20percent_validation_v2.json \
  --workers 15
```

**Retry failed extractions:**
```bash
python scripts/phase3_validate_vocabulary_concurrent.py \
  --poster_dir /path/to/posters \
  --id_mapping data/recbole/ml-1m/id_mappings.json \
  --vocabulary results/standard_entity_vocabulary_v2.json \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --output results/phase3_20percent_validation_v2.json \
  --workers 15 \
  --retry-errors
```

## Performance

### Expected Speed

| Dataset Size | Single-threaded | Multi-threaded (15 workers) | Speed-up |
|--------------|-----------------|----------------------------|----------|
| 5% (~200)    | ~20 min         | ~2 min                     | 10x      |
| 20% (~683)   | ~70 min         | ~5-7 min                   | 10-14x   |
| 80% (~3400)  | ~5-6 hours      | ~25-35 min                 | 10-14x   |

*Actual speed depends on API latency, network, and rate limits*

### Optimization Tips

1. **Find optimal worker count**: Start with 15, increase to 20 if no rate limits
2. **Monitor API rate limits**: If you see frequent rate limit errors, reduce workers
3. **Use faster network**: Cloud VM near API endpoint for lower latency
4. **Adjust save-interval**: Higher interval (20-30) reduces I/O overhead

## Differences from Single-threaded Version

| Feature                  | Single-threaded | Multi-threaded |
|--------------------------|-----------------|----------------|
| API calls                | Sequential      | Concurrent     |
| Progress update          | After each item | Real-time      |
| Save frequency           | After each item | Every N items  |
| Thread safety            | N/A             | Lock-protected |
| Speed (20% dataset)      | ~70 min         | ~5-7 min       |
| Memory usage             | Low             | Medium         |
| Recommended for          | Small tests     | Production     |

## Output Format

Same as single-threaded version:

```json
{
  "phase": "phase3_validation",
  "percentage": 20.0,
  "config": {
    "backend": "openai",
    "model": "gpt-4o-mini",
    "concurrent_workers": 15,
    "save_interval": 10,
    ...
  },
  "results": [...],
  "vocabulary_stats": {
    "total_knowledge_points": 6412,
    "valid_entities": 4950,
    "coverage_percentage": 77.2,
    ...
  }
}
```

## Troubleshooting

### Rate limit errors
- Reduce `--workers` (try 10 or 5)
- Add delays in API client if available

### Memory issues
- Reduce `--workers`
- Increase `--save-interval` to reduce I/O

### Inconsistent results
- Ensure `--temperature 0.0` for reproducibility
- Use same `--seed` for sampling

### Extraction hangs
- Check API connectivity
- Verify API key and base URL
- Look for error messages in output

## Migration from Single-threaded

To migrate existing workflow:

1. **No changes needed**: Drop-in replacement, same parameters
2. **Add concurrency**: Just add `--workers 15`
3. **Adjust save frequency**: Add `--save-interval 20` if needed
4. **Resume works**: Existing checkpoint files compatible

## Next Steps

After running Phase 3 validation with v2:

1. **Check coverage**: Look at `vocabulary_stats.coverage_percentage`
2. **Analyze NEW_ entities**: Use `scripts/analyze_new_entities.py` if needed
3. **Proceed to Phase 4**: If coverage ≥ 85%, run full 80% extraction
4. **Iterate**: If coverage < 85%, expand vocabulary to v3 or fix prompts
