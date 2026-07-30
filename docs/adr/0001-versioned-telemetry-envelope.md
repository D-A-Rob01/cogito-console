# ADR 0001: Versioned normalized telemetry envelope

Status: Accepted
Date: 2026-07-13

## Context

Cogito Console currently sends provider-specific legacy token packets. That format cannot represent capability snapshots, missing measurements, layer summaries, resource samples, or replay provenance uniformly.

## Decision

Add an append-only `TelemetryEvent` envelope with:

- semantic schema version;
- event ID, event type, session ID, run ID, and sequence;
- collection timestamp and optional token/layer indexes;
- status: `REAL`, `DERIVED`, `SYNTHETIC`, or `UNAVAILABLE`;
- source/provider/runtime and capability name;
- evidence IDs;
- bounded JSON payload.

Events are additive to the existing websocket packet during migration. The server persists them in a dedicated table and can export a bounded session trace as JSONL.

Full tensors are not stored. Token distributions become compact selected/top-k/full-distribution summaries; hidden states become last-token per-layer norms and deltas. Retention limits are enforced before persistence.

## Consequences

- every absence can be a queryable `measurement.unavailable` event;
- UI and storage can align token-, layer-, request-, and system-level evidence;
- older clients continue to consume legacy token packets;
- schema evolution requires migration tests and explicit version changes.
