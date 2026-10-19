# Archived training and testing evidence

`reference/<configuration>/seedN/` contains gzip-compressed original training text/scalar logs and an original evaluation summary. Local research, dataset and home paths are replaced by `${RESEARCH_ROOT}`, `${EGOHOS_ROOT}`, `${PRETRAINED_ROOT}` and `${HOME}`; numerical records, seeds and iteration counts are unchanged. `weights/manifest.json` links each record to the original and sanitized hashes. Multiple files may represent original/restarted segments; none are fabricated or concatenated to imply uninterrupted training.

`reference/` is historical experiment evidence. `release_checks/` contains new packaging/smoke-test evidence and must not be confused with full retraining. New training creates MMEngine logs and scalar JSON in `outputs/`; testing creates `summary.json` and per-image confusion JSONL. Validation is separate from the three-test-set Avg. The reference directory contains no image dumps or optimizer checkpoints.
