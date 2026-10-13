def recall_at_k(hits: list[bool], k: int) -> float:
    if any(hits[:k]):
        return 1.0
    else: 
        return 0.0

def reciprocal_rank(hits: list[bool]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1 / rank

    return 0.0