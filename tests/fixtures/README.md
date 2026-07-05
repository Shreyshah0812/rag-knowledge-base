# Test fixtures

The integration tests expect a `sample_policy.pdf` in this directory containing
known, checkable content (e.g. a fake employee handbook with a stated sick-day
policy, a vacation policy, and NOT containing anything about cryptocurrency
compensation).

Generate one quickly, e.g. with the `docx` or `pdf` skill available in your build
environment, or write a short mock policy doc in any word processor and export
to PDF. Keep it to 2-5 pages -- it only needs to exercise the pipeline, not be
realistic.

The integration tests are skipped automatically (not failed) if this file is
absent, so the test suite still runs cleanly without it.
