from scripts.review_sample import DOUBTFUL, RANDOM, sample


def rows(n_doubtful, n_fine):
    return ([{"question": f"d{i}", "judge_supported": "False"} for i in range(n_doubtful)]
            + [{"question": f"f{i}", "judge_supported": "True"} for i in range(n_fine)])


def test_sample_mixes_doubtful_and_random_answers():
    chosen = sample(rows(40, 200))
    assert len(chosen) == DOUBTFUL + RANDOM
    assert sum(r["judge_supported"] == "False" for r in chosen) == DOUBTFUL


def test_sample_works_with_few_doubtful_answers():
    chosen = sample(rows(3, 100))
    assert sum(r["judge_supported"] == "False" for r in chosen) == 3
    assert len(chosen) == 3 + RANDOM
