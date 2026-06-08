# Demo: route a chat request with local-first fallback

You run Ollama locally but want to transparently fall back to a self-hosted
vLLM box and then to cloud (OpenAI / Anthropic) if the locals are down — all
behind a single model alias. MODELROUTE plans that route offline so you can
inspect exactly which backend will be hit, the fallback order, the estimated
cost, and the concrete HTTP request each provider receives.

## Resolve the alias `fast` with a local-first strategy

```sh
python -m modelroute route fast --messages-file demos/01-basic/messages.json
```

Expected (table): the chain is ordered locals-first — `ollama-local` is rank 0
(cost $0.000000), then `vllm-host`, with cloud providers omitted because no API
keys are present.

## Include cloud providers and pick the cheapest

```sh
python -m modelroute --format json route fast \
  --messages-file demos/01-basic/messages.json --have-keys -s cheapest
```

Now `gpt-4o-mini` / cloud candidates appear in the chain, ranked by estimated
USD cost for the request. The `request` field shows the exact translated body.

## Simulate the primary backend going down

```sh
python -m modelroute --format json simulate fast \
  --messages-file demos/01-basic/messages.json --fail ollama-local
```

Dispatch falls back to `vllm-host`; the result reports `fell_back: true` and
the recorded error from the failed provider.
