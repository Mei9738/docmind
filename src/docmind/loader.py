from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    text: str
    metadata : dict[str,str]


def load_documents(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for f in sorted(directory.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        metadata = {
            "source": f.name
        }
        docs.append(Document(text=text, metadata=metadata))
    
    return docs

