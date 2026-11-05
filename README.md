# Atomberg Share-of-Voice Agent

An autonomous agent that measures how visible a brand is in search, and how well that visibility lands, against its competitors. Given a seed query such as "smart fan", it crawls up to 2,000 YouTube and Google results, detects brand mentions, runs document-level and aspect-level sentiment, and benchmarks Atomberg against 10 competitors on three metrics. The findings feed a Streamlit dashboard.

## How it works

```
            ┌──────────── budget not met, variants left ─────────────┐
            ▼                                                        │
 plan ──► crawl (one query variant: YouTube + Google) ──► router ────┘
                                                           │ budget met / exhausted
                                                           ▼
                                   analyze (mentions, sentiment, ABSA, metrics)
                                                           ▼
                                   insights (gaps, risks, rising competitors)
```

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) state machine (`src/sov/agent.py`):

- **plan** expands the seed into variants such as "best smart fan", "smart fan review" and "Atomberg smart fan", then splits the 2,000-result budget across them. Google Custom Search returns at most 100 results per query and YouTube search tails off after a few hundred, so one query alone can't reach the budget.
- **crawl** runs one variant per step on every configured platform and de-duplicates by URL. If one platform fails (quota, outage), the run continues.
- **router** keeps crawling until the budget is met, the variants run out, or three variants in a row add nothing new.
- **analyze** and **insights** score everything and write the findings.

If LangGraph isn't installed, a minimal runner executes the same nodes and edges, so the pipeline also runs in lightweight environments.

## Metrics

| Metric | What it answers | How |
| --- | --- | --- |
| **Mention Share (RMS)** | How often is each brand named? | Share of all brand mentions |
| **Weighted SoV (wSoV)** | How much attention does that visibility get? | Mentions weighted by engagement × sentiment |
| **Share of Positive Voice (SoPV)** | How much of the conversation is good news? | wSoV restricted to positive items |

Engagement is `0.6·log(1+views) + 0.3·log(1+likes) + 0.1·log(1+comments)`. Google results have no view counts, so their reach is estimated from search position using an organic click-through-rate curve. Without this, every Google row would silently drop out of wSoV and SoPV.

When an item mentions several brands, its weight is split in proportion to their mentions. Product names roll up to the parent brand, and "Atomberg Renesa" counts as one Atomberg mention, not two.

### Aspect-based sentiment

`src/sov/absa.py` extracts what people talk about (noise, energy, smart features, design, price, airflow, build quality, service) and the sentiment for each, per brand. Attribution is per sentence, so "Atomberg is silent but Havells is noisy" scores each brand separately. A sentence that names no brand inherits the brand of a single-brand document.

| Component | Full backend | Offline fallback |
| --- | --- | --- |
| Document sentiment | Hugging Face DistilBERT SST-2 | Review-tuned lexicon with negation and intensifiers |
| Aspect sentiment | PyABSA ATEPC (multilingual) | Aspect keyword map + clause-level lexicon |

Both default to `auto`: the transformer backend when installed, otherwise the lexicon. Set `sentiment.backend` and `absa.backend` in `config.yaml` to pin one.

## Quick start

```bash
make install            # core + agent + dashboard, lexicon NLP
make analyze            # score the sample crawl in data/ (no API keys needed)
make dashboard          # http://localhost:8501
```

To crawl live data, set the keys in `.env.example`, then run:

```bash
make install-full       # adds YouTube client, transformers, PyABSA
sov run                 # crawls config seeds, or: sov run --query "bldc fan" --budget 1000
```

| Command | Does |
| --- | --- |
| `sov run` | Agent: crawl, analyze, print and save insights |
| `sov analyze` | Re-score the CSVs in `data/` without API calls |
| `python -m sov.collectors.youtube --query ...` | Single-platform crawl |
| `python -m sov.collectors.google --query ...` | Single-platform crawl |

## Configuration

`config.yaml` holds the brands, with aliases and product lines, the 10 competitors, query modifiers, the crawl budget and the NLP backends.

## Outputs (`out/`)

| File | Contents |
| --- | --- |
| `summary.json` | RMS, wSoV and SoPV totals and shares, benchmark, sentiment breakdown, aspects, top publishers |
| `insights.json` | Ranked findings: risks, gaps, rising competitors, strengths |
| `brand_summary.csv` | One row per brand |
| `aspects.csv` | Brand × aspect sentiment |
| `scored.csv` | Every item with mentions, sentiment and engagement |

## Development

```bash
make test
```

The tests cover brand matching, the metrics (including NaN handling and weighting), both NLP backends, the collectors (against fake API clients), the agent's stopping rules, and an end-to-end analysis run. CI runs them on Python 3.10 and 3.12 and smoke-tests the CLI on the sample dataset.

```
src/sov/
  agent.py        LangGraph agent
  analyze.py      scoring pipeline
  metrics.py      RMS / wSoV / SoPV and benchmarking
  brands.py       alias-aware mention matcher
  sentiment.py    document sentiment
  absa.py         aspect-based sentiment
  insights.py     findings
  expand.py       query expansion and crawl budget
  collectors/     YouTube Data API and Google Custom Search
src/app.py        Streamlit dashboard
```
