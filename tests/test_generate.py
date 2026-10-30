from docmind.generate import NO_ANSWER, generate_answer


def test_generate():
    res = generate_answer("anything", [], None)

    assert res.text == NO_ANSWER
    assert res.sources == []