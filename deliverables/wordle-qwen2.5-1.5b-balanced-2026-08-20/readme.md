# Wordle Qwen2.5 1.5B balanced adapter

Development model produced on 2026-08-20 under the project's non-cheating
Wordle harness. This is the best multi-metric model from this experiment
cycle, not a claim of solved or production-grade Wordle performance.

## Files

- `wordle-qwen2.5-1.5b-balanced-adapter.zip`: PEFT LoRA adapter, adapter
  configuration, tokenizer files, and chat template.
- SHA-256 of ZIP:
  `e7e0259ec2369e0f6fbd2158f235bd24813005fd7b169dbafce325512c003a27`
- SHA-256 of the adapter safetensors inside:
  `ce77e5addf0f512178eeef16e1ab53455b3dfe8c860c14277e7f1d51f6af126f`

## Base model and training

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Base revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- License: Apache-2.0
- Adapter run: `sft-common-balanced-word-s2026-b2c325bec6`
- Curriculum: `COMMON-WORD-CURRICULUM-002`
- Data: 512 training-only, inference-shaped states from a public 128-word
  development universe; 96 train secrets and 32 held-out development secrets
- Training: 600 optimizer steps, seed 2026, learning rate 5e-5, LoRA rank 16,
  target-word loss weight 8, effective batch size 4
- Accounting: 357,905 optimizer tokens, 18,464,768 trainable parameters,
  175.7 seconds training time, 8.78 GB peak allocated VRAM

## Honest development results

The canonical greedy evaluation achieved:

| Metric | Previous Gemma 3 270M | This adapter |
| --- | ---: | ---: |
| Wins | 8/32 (25.0%) | 14/32 (43.75%) |
| Gameplay terminal compliance | 89.2% | 94.1% |
| Gameplay repeat rate | 27.4% | 2.0% |
| Diagnostic action-target accuracy | 14.1% | 41.3% |
| Turn-2 posterior violations | 79.3% | 35.1% |
| Singleton accuracy | 4/74 (5.4%) | 7/74 (9.5%) |

This is a small common-word development benchmark, not the full New York
Times Wordle distribution. The frozen 1,000-answer test was not opened.
Performance should not be reported as a general 43.75% Wordle win rate.

Several alternatives were rejected:

- targeted singleton curriculum: also 14/32 but much worse action accuracy
  and posterior consistency;
- repetition-penalty checkpoint: 16/32, but 20.6% invalid guesses and 7.2%
  repeats;
- deduplicated on-policy recovery: better singleton accuracy (15.4%) but only
  13/32 and worse overall constraint consistency;
- strict anchored continuation: 11/32 with worse gameplay compliance.
- matched Qwen2.5 3B: 15/32 final and 16/32 at step 450, but only 71.2%
  and 74.6% gameplay compliance, so both checkpoints were rejected.

## Non-cheating contract

The model receives only the normal Wordle instructions and the full visible
guess/feedback history. Generation is natural model generation. The harness
does not inject candidate words, mask the vocabulary, rerank or select a guess,
ban repeats, repair malformed output, or reveal the hidden answer. A valid
turn must end with exactly `Final answer: WORD`, where `WORD` is five ASCII
letters. Raw per-game logs and fixed-state diagnostics are retained in the
source run.

Protocol: `WORDLE-PROTOCOL-002`

Frozen protocol SHA-256:
`afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a`

## Loading

Extract the ZIP, download the exact base revision above, then load the adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = r"path\to\extracted-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(base, adapter_dir)
```

For exact reproduction, use the source repository's evaluator and greedy
decoder. Do not add constrained decoding and call it the same result.

## Recommended next research

The strongest remaining bottleneck is late-turn and singleton state recovery.
Before any frozen-test access, replicate the winning recipe over three seeds,
add an untouched final-development split, and require at least 99% terminal
compliance, under 30% turn-2 violations, and materially higher singleton
accuracy. A licensed Gemma 3 1B matched-family run would separate architecture
from capacity; this run is intentionally labeled a cross-family capacity
experiment.
