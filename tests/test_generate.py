from docmind.generate import generate_answer, NO_ANSWER

def test_generate():
    res = generate_answer("anything", [], None)

    assert res.text == NO_ANSWER
    assert res.sources == []