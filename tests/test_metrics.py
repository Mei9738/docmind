from docmind.metrics import recall_at_k, reciprocal_rank


def test_metrics():
    hits = [False, False, True, False]

    assert recall_at_k(hits, 3) == 1.0
    assert recall_at_k(hits, 1) == 0.0
    assert round(reciprocal_rank(hits), 2) == 0.33