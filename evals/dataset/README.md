# Golden evaluation set

`golden.yaml` — 64 questions stratified over the corpus:

| Type | Count | Tests |
|---|---:|---|
| `answerable_single_hop` | 44 | recall/precision/nDCG — one correct source |
| `answerable_multi_hop` | 4 | recall across two related sources |
| `off_topic` | 5 | abstention — plausible-sounding, outside the corpus |
| `conversational` | 4 | router sends these away from retrieval entirely |
| `multi_turn_followup` | 3 | query rewriting resolves a pronoun/reference |
| `adversarial` | 4 | guardrail blocks before retrieval or generation |

Two "benign" entries (`benign-delete-campaign`, `benign-block-users-survey`) are
folded into `answerable_single_hop` — they exist to catch the guardrail policy
over-triggering on ordinary product verbs ("delete", "block") that look
alarming out of context.

## This is a draft — review before trusting it

`expected_sources` were written from the corpus file listing and file-name
signal, not by reading every one of the 106 pages end to end. Treat the first
`make eval` run as **"which labels need fixing" as much as "how good is
retrieval."** A few likely failure points:

- **Multi-source entries** (e.g. `coachmark-definition` listing both the
  campaign-designs page and the glossary) may need trimming to whichever page
  actually defines the term, once you check.
- **Multi-hop entries** assume the two pages named are both necessary to fully
  answer the question — verify that's true rather than just plausible.
- **The exact question phrasing** may not match how a real user would ask;
  rephrase entries that feel unnatural once you've watched actual queries.

Fix labels by editing `golden.yaml` directly — there's no separate tool. After
material edits, `python evals/run_eval.py --save-baseline` regenerates
`evals/dataset/baseline.json` so the CI gate compares against the corrected
set, not the stale one.

## Extending it

Real user questions, once you have any, are worth more than more synthetic
ones — they use vocabulary and phrasing generated questions don't. Add them
under `answerable_single_hop` (or a new `type` if they don't fit) with the
document(s) that should have answered them.

Label at the **document** level (`expected_sources`), not the chunk level.
Chunk IDs are content-addressed and survive re-ingestion, but not a chunker
config change — document paths survive re-chunking, re-embedding, and
reindexing, which is exactly the kind of change you'll want to measure against
a stable baseline.

## Schema

```yaml
- id: unique-kebab-case-id
  question: "the question text"
  type: answerable_single_hop | answerable_multi_hop | off_topic |
        conversational | multi_turn_followup | adversarial
  expected_route: technical | conversational | off_topic | blocked
  should_abstain: true | false
  expected_sources: [repo-relative/posix/paths.md, ...]   # [] if none
  history:                                                 # multi_turn_followup only
    - role: user | assistant
      content: "..."
```
