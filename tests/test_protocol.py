from wordle_lab.protocol.env import WordleEnv, score_wordle
from wordle_lab.protocol.parsing import parse_terminal_answer
from wordle_lab.protocol.prompting import render_user_prompt
from wordle_lab.methods.rewards import shaped_reward
from wordle_lab.data.canonical import generate_canonical_states


def test_duplicate_scoring():
    assert score_wordle("APPLE", "ALLEY") == "GYBYB"
    assert score_wordle("AABCD", "AAAEE") == "GGBBB"


def test_terminal_parser_is_strict():
    allowed = ["CRANE", "SLATE"]
    assert parse_terminal_answer("analysis\nFinal answer: crane", allowed)["parsed_guess"] == "CRANE"
    assert parse_terminal_answer("Final answer: CRANE\nmore", allowed)["parsed_guess"] is None
    assert parse_terminal_answer("I choose CRANE", allowed)["parsed_guess"] is None
    assert parse_terminal_answer("Final answer: ABCDE", allowed)["status"] == "invalid_word"


def test_invalid_does_not_consume_turn():
    env = WordleEnv("CRANE", ["CRANE", "SLATE"])
    assert not env.step(None)["valid"]
    assert len(env.history) == 0


def test_prompt_replay_is_deterministic():
    history = [("SLATE", "BBGBG")]
    assert render_user_prompt(history) == render_user_prompt(history)


def test_reward_decomposition():
    value = shaped_reward(solved=True, information_gain=2, oracle_regret=0.5, repeated=False, format_valid=True)
    assert value["total"] == sum(value["components"].values())


def test_canonical_generator_can_share_a_public_answer_universe():
    rows = generate_canonical_states(
        ["CRANE", "SLATE"],
        "toy",
        target_count=3,
        seed=1,
        opener_count=2,
        answer_vocabulary=["CRANE", "SLATE", "PLANT", "APPLE"],
    )
    assert len(rows) == 3
    assert {row["secret_answer"] for row in rows} <= {"CRANE", "SLATE"}
