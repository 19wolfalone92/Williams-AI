# v7 mandatory edition/family retrieval profile

The historical v6 benchmark has one static question per book, a 3-card lexical RAG limit, and deterministically identical retrieved rules for all 25 charts in a book. This produced concrete omissions for NTD ENTRY and TC2 EXIT and a cross-edition wording concern in TC1.

Added candidate module williams_ai/v7_edition_family_retrieval.py. It is not imported by v6 and does not change frozen prompts, model weights, or outputs. It forces mandatory rule cards by edition+task family, verifies source edition and REVIEWED_PAGES status, appends up to three optional lexical cards, preserves exact PDF pages, and rejects unsupported TC1 Alligator wording. The tests assert mandatory NTD prereqs and TC2 reverse/WM1 coverage.

This is a source-selection safety improvement only. The profile has not been tested with real Qwen inference and does not establish that model reads or applies the cards correctly. Book-question correctness and actual market reasoning still require independent expert review. No gold labels, training or trading authority are included.
