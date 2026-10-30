FROM python:3.12

WORKDIR /docmind

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .

COPY src/ src/
COPY data/ data/
RUN pip install .

RUN python -m docmind.ingest
CMD ["uvicorn", "docmind.api:app", "--host", "0.0.0.0", "--port", "8000"]