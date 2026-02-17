# Test Bench Implementation Summary

## Overview

The test bench has been implemented with comprehensive scenarios covering all use-cases:
- **gender_en**: RoBERTa, DistilBERT, GPT-2 CLM
- **gender_de**: BERT baseline (no pruning), BERT with pruning, German GPT-2 + MLM head
- **race_religion**: DistilBERT for both race and religion bias

## First Run: Setting Expectations

For the first run, all tests are configured to:
1. **Print actual scores** with `[EXPECTATION SETTING]` prefix for easy identification
2. **Only check basic sanity** (not NaN, reasonable range)
3. **Allow you to review outputs** and set proper thresholds

After reviewing the first run outputs, you can:
- Update `DEFAULT_MIN_CORRELATION` and `DEFAULT_TOLERANCE` in `verify_utils.py`
- Replace the basic assertions with `assert_correlation_threshold()` calls
- Set model-specific thresholds if needed

## Updated Files

### `test_bench/verified/verify_utils.py`
- Updated `BENCH_TRAIN_CONFIG` with new parameters:
  - `eval_max_size=50` (was 20)
  - `n_evaluation=250` (was 50)
  - `max_iterations=1000` (was 500)
  - `learning_rate=1e-5` (was 1e-4)
- Added `BENCH_TRAIN_CONFIG_WITH_PRUNING` (includes pre/post pruning)
- Added `BENCH_MLM_HEAD_CONFIG` (for decoder-only MLM head training)

### `test_bench/verified/test_gender_en.py`
- **test_gender_en_roberta_verified**: RoBERTa-base (encoder-only, different MASK token)
- **test_gender_en_distilbert_verified**: DistilBERT (encoder-only, standard [MASK])
- **test_gender_en_gpt2_clm_verified**: GPT-2 (decoder-only, direct CLM)

### `test_bench/verified/test_gender_de.py`
- **test_gender_de_bert_baseline_verified**: BERT baseline without pruning (reference)
- **test_gender_de_bert_with_pruning_verified**: BERT with pre/post pruning (compares to baseline)
- **test_gender_de_gpt2_mlm_verified**: German GPT-2 + MLM head workflow

### `test_bench/verified/test_race_religion.py`
- **test_race_religion_distilbert_verified**: Tests both race (white/black) and religion (christian/muslim) scenarios

## Running Tests

### Run all test bench tests:
```bash
python test_bench/run_bench.py
```

### Run specific test file:
```bash
pytest test_bench/verified/test_gender_en.py -v -m integration
```

### Run single test:
```bash
pytest test_bench/verified/test_gender_en.py::test_gender_en_roberta_verified -v -m integration
```

## Expected Output Format

Each test will print:
```
[EXPECTATION SETTING] <test_name>: score=<actual_score>
```

Example:
```
[EXPECTATION SETTING] gender_en (RoBERTa): score=0.6234
[EXPECTATION SETTING] gender_en (DistilBERT): score=0.5891
[EXPECTATION SETTING] gender_de BERT baseline (no pruning): score=0.7123
[EXPECTATION SETTING] gender_de BERT with pruning: score=0.6987
  Baseline score: 0.7123, Pruned score: 0.6987, Difference: 0.0136
```

## Next Steps After First Run

1. **Review all scores** from the `[EXPECTATION SETTING]` output
2. **Set appropriate thresholds** based on actual results:
   - Update `DEFAULT_MIN_CORRELATION` if needed
   - Consider model-specific thresholds
   - Set pruning degradation tolerance
3. **Replace basic assertions** with `assert_correlation_threshold()` calls
4. **Add comparison logic** for pruning vs baseline tests

## Test Configuration Summary

| Test | Model | Config | Pruning |
|------|-------|--------|---------|
| gender_en_roberta | roberta-base | encoder-only | Yes |
| gender_en_distilbert | distilbert-base-cased | encoder-only | Yes |
| gender_en_gpt2_clm | gpt2 | decoder-only CLM | Yes |
| gender_de_bert_baseline | bert-base-german-cased | encoder-only | No (baseline) |
| gender_de_bert_pruned | bert-base-german-cased | encoder-only | Yes |
| gender_de_gpt2_mlm | dbmdz/german-gpt2 | decoder-only + MLM | Yes |
| race_religion | distilbert-base-cased | encoder-only | Yes |
