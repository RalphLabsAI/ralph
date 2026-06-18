# Running a Ralph validator (subnet 40)

Ralph validators run the **GPU** verification. Each epoch a validator pulls miner submissions, runs the
eval ladder + held-out hidden-eval, scores them, sets weights, and publishes a **signed audit report
anchored on-chain** — so anyone can verify it on CPU (see [`auditor/README.md`](../auditor/README.md)).

## What it does each epoch
1. Pull new submissions (proof bundles).
2. Verify: patch/integrity → attestation → training-log sanity → held-out **hidden-eval** (val-BPB + benchmark) vs the king.
3. Decide: a submission that decisively beats the king (past the measured **0.013** val-BPB noise floor) becomes the new king.
4. Set weights on-chain.
5. Publish a signed audit report + anchor its hash on-chain.

## Hardware
A **GPU** (H100 / H200) for the hidden-eval. Miners pay the training cost — the validator only runs eval. CPU works but is slow.

## Install
```bash
git clone https://github.com/RalphLabsAI/ralph.git
git clone https://github.com/RalphLabsAI/recipe.git     # canonical model, side-by-side
cd ralph && python -m venv .venv && source .venv/bin/activate
pip install -e '.[chain]'                                # bittensor>=10, python>=3.10
```

## Register on netuid 40
```bash
btcli subnet register --netuid 40 --network finney --wallet.name <wallet> --wallet.hotkey <hotkey>
```

## Configure — `.env` in the repo root
```bash
RALPH_CHAIN=bittensor
BT_NETWORK=finney
BT_NETUID=40
BT_WALLET=<wallet>
BT_HOTKEY=<hotkey>
BT_WALLET_PASSWORD=<coldkey-password>
HF_TOKEN=<hf-token>
RALPH_HF_REPO=RalphLabsAI/proof-bundles
```

## Run
```bash
set -a && source .env && set +a
python -m validator.service \
  --epoch-seconds 120 \
  --noise-floor 0.013 \
  --hf-repo RalphLabsAI/proof-bundles --hf-token $HF_TOKEN \
  --hf-publish-audit                       # publish signed audit reports to RalphLabsAI/audit-reports
```
It polls submissions, scores them, sets weights, and publishes an audit report each epoch.

## Get audited
Anyone can independently check your validator on CPU — no GPU, no re-doing the eval. Point them at
[`auditor/README.md`](../auditor/README.md): `python -m auditor --once`.

## Links
- Code: github.com/RalphLabsAI/ralph · github.com/RalphLabsAI/recipe
- On-chain: taostats.io/subnets/40
- Discussions: github.com/RalphLabsAI/ralph/discussions
