# Datasets

No dataset is bundled and nothing is downloaded automatically. Record the source,
version, licence and date of every dataset you use here, because the viva/report
needs this provenance.

Expected canonical CSV (any CSV/JSON/HF dataset can be mapped to it via `app/data.py`):

    text,label,source,generator,domain
    "…",0,"<where it came from>","human","news"
    "…",1,"<where it came from>","gpt-4","news"

`label`: 0 = human, 1 = AI. `generator` should name the model that wrote AI rows
(and e.g. "human" for human rows) so unseen-generator splits are possible.
`domain` (news, essays, reviews, ...) enables unseen-domain splits.

Candidate public sources to evaluate (verify availability, licences and exact
column names yourself before use; none of this is claimed to be suitable
without checking):
- RAID benchmark (Dugan et al., 2024) - multiple generators, domains, attacks
- HC3 (Human ChatGPT Comparison Corpus)
- Other human-vs-LLM corpora cited in the surveys listed in the main README

Dataset log (fill in):

| name | source URL | version/date | licence | rows | notes |
|------|-----------|--------------|---------|------|-------|
