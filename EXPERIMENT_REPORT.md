# WORDLE-PROTOCOL-002 build and pilot report

Date: 2026-08-14

## Status

The experiment infrastructure and development pilots are operational. All
trained-model results below are pipeline/dose pilots, not primary Stage-1
evidence: they use one seed and 25 development answers. No trained checkpoint
has accessed the locked 1,000-answer test. The full base-model test baseline is
the only unconditional test evaluation authorized by the design.

The authoritative frozen protocol hash is
`afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a`.

## Built

- strict `Final answer: WORD` final-line parser with no recovery or repair;
- shared greedy 128-new-token decoder and six-valid-turn environment;
- 2,048 train and 512 dev canonical states with four rendered views;
- direct and rationale preference views with an exact 80/10/10 negative mix;
- test-answer leakage checks, replay/parser/reward tests, and component hashes;
- content-derived run IDs, resumable status states, dev/test selection gate;
- LoRA SFT with completion-only masking and 25/50/75/100% checkpoints;
- DPO, ORPO, and GRPO backends plus decomposed shaped reward;
- model-call, token, latency, VRAM, parameter, trace, and strategy diagnostics;
- paired-bootstrap, exact McNemar, Holm-correction, tidy collection, and chart
  foundations.

## Development findings

| Condition | States / steps | Dev wins | Terminal compliance | Main behavior |
|---|---:|---:|---:|---|
| Base Gemma | 0 / 0 | 0/25 | 0% | Emits literal placeholder `WORD` |
| Direct SFT | 512 / 100 | 0/25 | 100% | Learns `ARLES`, repeats it |
| Episode multi-turn | 512 / 100 | 0/25 | 100% | Same repeat collapse |
| Mixed curriculum | 512 / 100 | 0/25 | 100% | Same repeat collapse at higher cost |
| Visible rationale | 512 / 100 | 0/25 | 100% | Fluent but unfaithful traces; repeats `ARLES` |
| Direct SFT, larger dose | 2,048 / 300 | 0/25 | 60.2% | Diversifies, but misspells/out-of-list/repeats |

The constraint-aware random-posterior and greedy-oracle references each won
25/25 development games; a true uniform random-allowed-word policy won 0/25.
The base retention score was 30%; the 300-step direct adapter fell to 0%.
Visible-rationale trace-fact accuracy was 11.1% overall: posterior count 0%,
fixed positions 16.7%, and excluded letters 16.7%.

The most useful causal decomposition so far is:

1. SFT readily teaches the terminal envelope (formatting).
2. Explicit root-state exposure teaches a valid fixed opener.
3. More state coverage produces some action diversity.
4. The model still fails state-conditioned action spelling/selection and loses
retention, so a Stage-2 winner comparison is not yet scientifically valid.

The frozen base test baseline won 0/1,000 with 0% terminal compliance; the
required 975-answer sensitivity slice was identical. On all 1,000 test answers,
the uninformed random-allowed reference won 0 and the greedy oracle won 998
(99.8%; 2 failures after six guesses).

## Gate decision

Do not select a representation, run DPO/ORPO/GRPO as a claimed method screen,
or open the trained-model test gate yet. The next development experiment should
test a common-word/action-curriculum or substantially larger SFT dose while
preserving the frozen evaluator and reporting the altered training signal. Only
after three seed-matched SFT parents show nonzero strategic performance should
the preregistered Stage-2 comparison begin.

## Follow-up intervention iteration

Date: 2026-08-14

The requested follow-up reproduced the repeated-word collapse and isolated four
causes that were conflated in the original pilots:

1. The 2,500-answer universe was a random subset of the 14,855-word legal-guess
   list, not a normal Wordle answer list. It contains obscure labels such as
   `GADJO`, `OXLIP`, `BUROO`, and `YITIE`.
2. The training oracle ranked actions against only 1,250 training secrets while
   evaluation diagnostics used all 2,500 answers. Identical visible histories
   therefore had different policy semantics across train and evaluation.
3. Oracle targets were extremely sparse: most later-turn target words occurred
   once. A 270M model learned the terminal prefix and one root action (`ARLES`)
   before it learned the state-to-action mapping.
4. The mixed failure curriculum corrected only a malformed placeholder
   (`XXXXX`); it did not demonstrate correction of the observed failure, a prior
   repeated guess. The episode representation was also trained on interleaved
   messages but evaluated through the flattened prompt.

The base prompt's literal `Final answer: WORD` example separately explains why
the unfine-tuned model emitted the placeholder `WORD`. This is prompt imitation,
not strategic behavior.

### Inference-only screen on the collapsed adapter

Four prompt variants and four decoders were screened on five development games.
Temperature 0.3 and 0.7 left the peaked `ARLES` mode unchanged. Matching the
episode model to its native interleaved prompt plus a 1.05 repetition penalty
created four unique guesses and zero measured repeats, but 73.3% of calls were
invalid and 68.8% of valid guesses violated the posterior. That condition was
rejected: it traded one failure for another.

### Common-word curriculum

`COMMON-WORD-CURRICULUM-001` ranks the pinned allowed list by English Zipf
frequency, fixes one public action/answer universe across train and development,
holds episode secrets out, renders compact exact feedback constraints, adds
prior-repeat correction examples, and floors exposure for rare root/late turns.
The evaluator still accepts unconstrained model text, uses the same strict final
line parser, and never masks logits, supplies candidate words, selects an action,
or repairs output.

| Condition | Universe | Dev wins | Compliance | Repeat rate | Unique guesses | Decision |
|---|---:|---:|---:|---:|---:|---|
| Base-start SFT, step 100 | 512 | 0/25 | 75.9% | 67.7% | 8 | too diffuse |
| Base-start SFT, step 400 | 512 | 0/25 | 56.3% | 58.0% | 16 | reject |
| Base-start SFT, step 150 | 128 | 0/25 | 100% | 58.7% | 9 | underdosed |
| Base-start SFT, step 300 | 128 | 1/25 | 98.7% | 43.9% | 26 | first win |
| Base-start SFT, step 450 | 128 | 1/25 | 100% | 41.8% | 34 | viable |
| Base-start SFT, step 600 | 128 | 1/25 | 100% | 39.7% | 29 | viable parent |
| Step 600 + repetition penalty 1.05 | 128 | 1/25 | 100% | 18.5% | 41 | accepted decoder |
| SFT + ORPO step 25 + penalty 1.05 | 128 | 1/25 | 100% | 10.1% | 42 | viable |
| SFT + ORPO step 50 + penalty 1.05 | 128 | 1/25 | 100% | 10.3% | 43 | selected dev dose |
| SFT + ORPO step 75 + penalty 1.05 | 128 | 1/25 | 93.5% | 18.2% | 45 | overdosed |
| SFT + ORPO step 100 + penalty 1.05 | 128 | 1/25 | 79.4% | 10.1% | 53 | reject |

The held-out win was real state-conditioned play: for secret `SENSE`, the model
guessed `SHARE`, received `GBBBG`, then emitted `SENSE` for all green on turn 2.

Temperature was not the best lever after training. At temperature 0.7, diversity
rose to 52 words and repeats fell to 26.7%, but the win disappeared and compliance
fell to 98.7%. Repetition penalty 1.10 cut repeats to 4.8% but introduced 4.6%
invalid calls. Penalty 1.05 retained perfect compliance and the win.

### Scaling boundary

Direct base-to-512 SFT and 128-to-512 continued SFT both failed. The continued
run had terminal compliance of 19.2%, 9.5%, 0%, and 60.6% at steps 100, 200, 300,
and 400 respectively, with no wins. The result argues for a gradual 128-to-256
curriculum and a capacity ablation rather than additional u512 dose.

### Updated decision

The repeated-word collapse is fixed in a bounded development regime: the best
clean condition uses 43 unique guesses with a 10.3% repeat rate, 100% strict
validity, and one held-out win. This is not evidence of full Wordle competence:
constraint violations remain 82.2%, the development sample is 25 games, and only
one seed has been run. Do not open the locked protocol-002 test gate.

Next work should be, in order:

1. repeat the u128 SFT parent and 50-step ORPO dose for three seeds;
2. add constraint-violation negatives, not only prior-repeat negatives;
3. scale through u256 before u512 and require compliance to remain above 99%;
4. compare the 270M model with a roughly 1B capacity condition under the same
   curriculum and token budget;
5. only then return to the preregistered method comparison.
