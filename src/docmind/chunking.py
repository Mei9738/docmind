from dataclasses import dataclass
from docmind.loader import Document

@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int]

def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")
    
    pieces : list[str] = []
    start = 0
    gap = size - overlap

    while True:
        pieces.append(text[start: start+size])
        if start + size >= len(text):
            break
        start = start + gap

    return pieces

def _save_section(chunks: list[Chunk], lines: list[str], source: str, heading: str,  size: int, overlap: int) -> None:
    text = "\n".join(lines)
    if not text.strip():
        return
    for piece in _split_text(text, size, overlap):
        chunks.append(Chunk(text=piece, metadata={
            "source": source,
            "heading": heading,
            "position": len(chunks) + 1,
        }))

def chunk_document(doc: Document, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    lines : list[str] = []
    in_code = False

    for line in doc.text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if line.startswith("#") and not in_code:
            _save_section(chunks, lines, doc.metadata["source"], heading, size, overlap)

            lines.clear()
            heading = line.lstrip("# ").partition(" { #")[0]
        else:
            lines.append(line)
    
    _save_section(chunks, lines, doc.metadata["source"], heading, size, overlap)
   
    return chunks

