# Cogito Console Demo Script

## 1. Start The App

```powershell
cd outputs\cogito-console
python -m uvicorn cogito_console.server:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## 2. Start A Run

Use the default prompt or enter a new one. Click **Start**.

Point out:

- The provider badge in the left panel.
- Prompt checks running before tokens appear.
- The session context status moving from routing to active.

## 3. Watch Prompt Checks

In the graph, prompt-check nodes appear before token streaming.

Say:

> These confidence values are synthetic local heuristics in demo mode. The point is not that they are real model internals; the point is that they are labeled and inspectable.

## 4. Watch Tokens Stream

The accepted path renders as the central token route. Alternatives branch out beside each token.

Point out:

- Token probability/status badge.
- Alternative probability/status badge.
- Signal panel values marked `SYNTHETIC`.

## 5. Hover An Alternative

Hover a ghost path.

Show:

- Exact probability/logprob.
- Status badge.
- Evidence ID.
- Plain explanation.
- Caveat.

## 6. Click An Alternative

Click **Use this path** or click the ghost node.

Expected result:

- The old path collapses grey.
- The selected branch becomes the new origin.
- The event timeline says the path changed.
- The stream resumes from the rewritten token.

## 7. Inspect A Token

Click a streamed token.

Show:

- "Why this number?"
- "Where did this come from?"
- Evidence ID.
- Expert Mode raw token event.

## 8. Inspect A Graph Edge

Use the graph and inspector to explain:

- Provider -> token means `generated_by`.
- Token -> alternative means `alternative_to`.
- Rewrite -> token means `caused_rewrite`.

## 9. Compare Branches

After steering, scroll to **Compare branches**.

Show:

- Before text.
- After text.
- Changed token.
- Old run ID.
- New run ID.
- Alternative metric if available.

Use this line:

> You changed the path here. The system restarted from that point with the selected token.

## 10. Show Evidence Badge

Turn on **Expert mode**.

Open raw data and show that the packet includes:

- `token_event`
- `metrics`
- `evidence`
- `raw_provider_payload`

Close with:

> The impressive part is not that every value is real. The impressive part is that every value tells you whether it is real.
