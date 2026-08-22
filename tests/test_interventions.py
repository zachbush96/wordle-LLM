from wordle_lab.experiments.intervention_sweep import PROMPT_VARIANTS


def test_explicit_prompt_names_prior_guess_and_expands_feedback():
    messages = PROMPT_VARIANTS["explicit_feedback"]([("CRANE", "GYBBB")])
    text = "\n".join(message["content"] for message in messages)
    assert "CRANE" in text
    assert "Forbidden repeats" in text
    assert "fixed [1=C]" in text
    assert "required [Cx1, Rx1 not@2]" in text
    assert "absent [A, E, N]" in text


def test_episode_prompt_replays_assistant_turns():
    messages = PROMPT_VARIANTS["native_episode"]([("SLATE", "BBGBG")])
    assert messages[-2] == {"role": "assistant", "content": "Final answer: SLATE"}
    assert "Do not repeat SLATE" in messages[-1]["content"]
