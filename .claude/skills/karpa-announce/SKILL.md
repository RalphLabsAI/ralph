---
name: karpa-announce
description: Draft paired community-discussion posts for a Karpa event — one GitHub Discussion (long-form, receipts-dense) and one Twitter / X post (short, headline-grade). Both stay LOCAL (gitignored); user reviews and publishes. Use when a publicly-announceable Karpa event lands (new king, major patch, experiment results, protocol-version bump, or follow-up to a prior post).
---

# /karpa-announce — paired community posts for Karpa events

## When to invoke

Whenever a publicly-announceable Karpa event lands:

- **king_change** — a new king crowned on the canonical chain
- **patch** — a major PR merged into `karpaai/karpa` or `karpaai/recipe`
- **test_run** — an experiment / methodology run with publishable results (typically lives in `karpa/experiments/<date>-<slug>/`)
- **protocol_bump** — version bump (v0.10 / v0.11 / v1.0) or new spec landing
- **followup** — follow-up to a prior post (by slug or post number)

Invocation: `/karpa-announce <event-type> [free-text context]`

Examples:
- `/karpa-announce test_run 2026-06-proxy-validation`
- `/karpa-announce king_change recipe-v0.2.0`
- `/karpa-announce followup 09_king_criteria_audit_and_experiment`

## What this skill produces

For every invocation, draft **two** files in lock-step with a matching slug:

1. **GitHub Discussion** at `karpa/github_discussions/karpaai_repo/NN_<slug>.md`
2. **Twitter / X post** at `karpa/twitter_posts/karpaai_account/NN_<slug>.md`

Both files stay LOCAL — `twitter_posts/` and `github_discussions/` are gitignored. The user reviews each file, then publishes manually.

## Step-by-step process

When invoked, do all of the following in order:

### 1. Find the next file numbers

```bash
ls /workspace/unicorn/karpathian/karpa/twitter_posts/karpaai_account/ | grep -oE '^[0-9]+' | sort -n | tail -1
ls /workspace/unicorn/karpathian/karpa/github_discussions/karpaai_repo/ | grep -oE '^[0-9]+' | sort -n | tail -1
```

Increment each by 1. The Twitter number continues from existing posts; the GH number is its own sequence (starts at 01).

### 2. Read one recent reference post for voice

To recalibrate voice each time, read one of: `02_phase0_mvp.md`, `07_deepdive_5_4_single_tier.md`, or `08_deepdive_5_7_antigaming.md` in `twitter_posts/karpaai_account/`. Pick whichever is structurally closest to the current event type.

### 3. Gather the receipts

For each event type, BEFORE drafting:

- **king_change** — read `chain/king.json`, `chain/events.jsonl` (latest entries), the king bundle's `submission.json`, the rationale.md inside the bundle. Identify: prior king's val_bpb, new king's val_bpb, delta, axis touched, miner hotkey (display name if known).
- **patch** — `git log -p <commit>` for the change, the PR description if available, any test additions.
- **test_run** — read the `verdict.json`, `summary.md`, and `PRE_REGISTERED.md` (if it exists) in the experiment dir. Pull the headline number + the key per-recipe table.
- **protocol_bump** — read the spec doc (`docs/`), the changelog entry, the migration plan.
- **followup** — read the prior post being followed up + the new data that's the reason for the follow-up.

### 4. Apply guardrails (mandatory)

Before drafting, mentally scan the planned content against this list. If any item leaks in, STRIP IT before writing:

**ABSOLUTELY DO NOT include:**
- References to private mainnet-simulation rounds (anything we ran on our own local validator with our own hotkeys)
- Names of specific hotkeys we operate (5F6W…, 5F23…, etc.)
- Cloud-provider rental specifics for private runs (Shadeform instance IDs, IPs, exact private-spend numbers)
- Private validator/miner setup details
- Any agent-A / agent-B / our-sim-rounds framing
- **Internal version labels** (v0.10, v0.11, v1.0, sprint numbers, phase numbers — these are project-management artifacts, not external research discussion)
- **Shipping timelines / release-schedule language** ("next 30 days", "in 8 weeks", "shipping Q4", "by end of month", "next quarter") — external posts discuss the protocol's *direction*, not when artifacts ship. Direction is durable; ship dates change.
- **Project-management framing in general** ("we'll ship", "MVP", "roadmap", "milestone") — these read as a corporate update. Use research framing instead ("the next version of the rule does X", "the protocol is moving toward Y", "going forward, the gate measures Z")

**ALLOWED references:**
- The public Karpa protocol (v0.9.2 / v0.10 / v0.11 / v1.0)
- Public on-chain events on netuid 16 visible via taostats.io
- Public PRs on `karpaai/karpa` and `karpaai/recipe`
- Public HF dataset `karpaai/proof-bundles`
- Public experiments in `karpa/experiments/` (the dir is intended to be public)
- The published Karpa whitepaper sections

If in doubt: a fact is safe if it's traceable to a public-repo file path, a public-chain transaction, or a publicly-published dataset. Otherwise skip it.

### 5. Draft the Twitter post first

It's the harder constraint. Iterate until it fits.

- **Short form** (default): ≤ 280 characters. Single tight sentence with the headline + 1 number + 1 link + tags.
- **Long form** (X Premium, when there's substantive content): ≤ 3500 chars; target 1500–2500.

Voice — researcher / dev discussion:
- Honest engineer, plain past tense ("we ran X", "Y landed", "Z surprised us")
- Specific numbers, receipts > claims
- No emoji
- No "we're excited to announce" / "stay tuned" / "game-changing" / token talk
- Treat the reader as a peer

Hashtag policy — Twitter only:
- 2–3 tags MAX per post, at the end on their own line
- Researcher-relevant only: `#LLM #pretraining #Bittensor`
- Swap in `#opensource` / `#scalinglaws` / `#attention` / `#optimizer` / `#reproducibility` when topical
- NEVER: `#crypto`, `#web3`, `$TAO`, `#1000x`, `#gem`, or any reach-bait

### 6. Expand to the GitHub Discussion post

Goal: 600–1500 words. Use code blocks, tables, links, evidence.

Structure:
```markdown
# <post title>

**Category:** Announcements | Show and tell | Q&A | Research | Ideas
**When to post:** <relative timing>
**Length:** ~<N> words
**Linked artifacts:** <PR / commit / experiment dir / HF dataset / Twitter post slug>

---

## Post body

<the actual GH Discussion markdown post>
```

Sections inside the post body (adapt to event type):
- **Headline finding / what changed** — first paragraph, the bottom line
- **Method / what we did** — code refs, data refs, pre-registration if applicable
- **Results** — table or numbers, with links to raw data
- **What we learned** — the interpretation, qualified honestly
- **What's next** — a concrete next step or open question for the community

Voice rules are identical to Twitter, but the constraint is different — receipts can be detailed, code blocks expected, tables welcome.

### 7. Write both files

Use the file format below.

Twitter file (`twitter_posts/karpaai_account/NN_<slug>.md`):
```markdown
# <topic title — concrete, no hype>

**When to post:** <day / hour after prior post / link to prior post>
**Length:** <N> chars (short | X Premium long-form)
**Attachments:** <none | [INSERT: description of screenshot]>
**Hashtags:** #tag1 #tag2 #tag3

---

## Post body

<the actual post text in a code fence so it's copy-paste ready>
```

GitHub Discussion file (`github_discussions/karpaai_repo/NN_<slug>.md`): per structure in step 6.

### 8. Update the GH Discussions index

Append a row to `github_discussions/README.md` index table:
```
| NN | <slug> | `twitter_posts/karpaai_account/NN_*` | ⬜ |
```

### 9. Report to the user

Surface in one message:
- Both file paths
- Headline of the Twitter post (so they see the framing)
- Headline of the GH Discussion (so they see the framing)
- Any guardrail flags that were close to triggering (so they can second-guess)

## Length caps (HARD)

- Twitter short: ≤ 280 chars
- Twitter long-form (X Premium): ≤ 3500 chars, target 1500–2500
- GH Discussion: 600–1500 words

If a draft is longer, fold detail into linked artifacts rather than the body.

## Voice reference posts

To match tone, read one of these (in `twitter_posts/karpaai_account/`):

- `02_phase0_mvp.md` — clean past-tense announcement, 1535 chars
- `07_deepdive_5_4_single_tier.md` — technical deep-dive, 3733 chars
- `08_deepdive_5_7_antigaming.md` — adversarial / mechanism-design, 4903 chars
- `09_king_criteria_audit_and_experiment.md` — pre-registered experiment, two-variant (announcement + verdict)

## Templates per event type

### king_change

**Twitter (short, ≤ 280):**
> New king on netuid 16. recipe-vX.Y.Z lands val_bpb=A.AA on axis "<axis>". Bundle 0x… (HF PR #N). Δ −0.0XX vs prior king at recipe-vX.Y.Z-1. Hypothesis: <one-line>. [HF link] [tags]

**GH Discussion:**
- Headline finding (the val_bpb delta and the axis)
- Hypothesis (what the patch tried to test)
- Receipt (recipe PR link, HF PR link, on-chain block #)
- Diff highlights (smallest diff that captures the change)
- What this opens (the next obvious axis to attack)

### patch

**Twitter (short or long):**
> Karpa <component> change merged: <one-sentence what>. Why it matters: <one-sentence>. PR: karpaai/<repo>#NNN. Tests: <N new>. [tags]

**GH Discussion:**
- What changed (the diff in plain language)
- Why (the bug / opportunity / spec gap that motivated it)
- How it's tested
- What it doesn't fix yet (open follow-ups)

### test_run

**Twitter (long-form, X Premium):**
- 1-paragraph headline finding
- 2-paragraph method (pre-registration → execution → result)
- 1-paragraph what we'd do differently next time
- Tags

**GH Discussion:**
- Question being answered
- Pre-registered decision rule (link to PRE_REGISTERED.md)
- Recipe set + configs (table)
- Headline numbers (table)
- Failure-mode discussion (the specific recipe inversions, with citations to literature where relevant)
- Action taken / proposed
- Raw data link

### protocol_bump

**Twitter (long-form, X Premium):**
- 1-paragraph what changed
- 1-paragraph why now (the bug or gap being fixed)
- Migration notes (1–2 sentences)
- Link to spec
- Tags

**GH Discussion:**
- Diff at the spec level (before vs after)
- Migration path for miners + validators
- Backward compatibility commitments
- Acknowledged residual limits ("what this still cannot see")
- Open invitation for feedback

### followup

**Twitter (short, ≤ 280):**
> Following up on <link to prior post>. New data: <one-sentence headline number>. The implication: <one-sentence>. [tags]

**GH Discussion:**
- Quote the prior claim / pre-registration
- New evidence
- How the new evidence updates the prior claim
- What we'd want next

## Don't do this (anti-checklist)

- Don't post the same content to both surfaces verbatim. Twitter is the headline; GH is the receipt. Different artifacts.
- Don't open with "we" — open with the finding.
- Don't draft both files in a hurry — voice misses cost trust.
- Don't include any hashtag in the GH Discussion.
- Don't commit either file to git (both dirs are gitignored).
- Don't reference the user by name in either draft.
- Don't fabricate numbers. If the experiment didn't run, say so. The protocol's credibility is downstream of every claim being verifiable.
