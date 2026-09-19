import os
import re
import json
import fitz  # PyMuPDF

INPUT_DIR = "./documenti_iva_oss"  # Folder where your PDFs are located
TEMP_DIR = "./temp"
OUTPUT_JSON = os.path.join(TEMP_DIR, "chunks_dataset.json")

os.makedirs(TEMP_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDFs respecting column layout and removing headers/footers."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        blocks = page.get_text("blocks")
        # Sort blocks based on vertical (y0) and horizontal (x0) position
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        for b in blocks:
            block_text = b[4].strip()
            # Filter out page headers or isolated footers
            if len(block_text) > 15 and not re.match(r'^\d+\s*$', block_text):
                full_text.append(block_text)
                
    return "\n\n".join(full_text)

def split_into_semantic_chunks(text, filename):
    """Splits text by logical units (Articles, Chapters, or Sections)."""
    chunks = []
    
    # Pattern to identify the start of a regulatory article or section
    # Handles both Italian (Art. / Articolo) and English (Article / Section)
    pattern = r'(\n(?:Articolo|Art\.|Article|CAPO|TITOLO|SECTION|Section)\s+\d+[\w\s\.-]*)'
    
    split_docs = re.split(pattern, text)
    
    if len(split_docs) <= 1:
        # Fallback if the document does not use the "Articolo" format: splits into large paragraphs (~1200 chars)
        raw_paragraphs = text.split("\n\n")
        current_chunk = ""
        for p in raw_paragraphs:
            if len(current_chunk) + len(p) < 1200:
                current_chunk += p + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        # Reconstructs sections by pairing the title with the content
        first_chunk = split_docs[0].strip()
        if first_chunk:
            chunks.append({"header": "Introduzione / Premessa", "body": first_chunk})
            
        for i in range(1, len(split_docs), 2):
            header = split_docs[i].strip()
            body = split_docs[i+1].strip() if i+1 < len(split_docs) else ""
            chunks.append({"header": header, "body": body})

    # Formats final chunks with metadata from the source document
    structured_chunks = []
    lang = "en" if "Notes" in filename or "Guidelines" in filename else "it"
    
    for idx, c in enumerate(chunks):
        if isinstance(c, dict):
            content = f"{c['header']}\n{c['body']}"
            section_title = c['header']
        else:
            content = c
            section_title = f"Sezione {idx+1}"

        if len(content.strip()) < 40:  # Skip empty or overly short chunks
            continue

        structured_chunks.append({
            "chunk_id": f"{os.path.basename(filename)}_{idx}",
            "source": filename,
            "language": lang,
            "section": section_title,
            "text": content
        })

    return structured_chunks

def run_pipeline():
    all_chunks = []
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    
    print(f"Found {len(pdf_files)} PDFs in the folder. Starting processing...")

    for file_name in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, file_name)
        print(f" Processing: {file_name}...")
        
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = split_into_semantic_chunks(raw_text, file_name)
            all_chunks.extend(chunks)
            print(f"  -> Generated {len(chunks)} semantic chunks.")
        except Exception as e:
            print(f"  -> ERROR on {file_name}: {e}")

    # Save to the temp folder
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nPreprocessing completed! Total chunk generated: {len(all_chunks)}")
    print(f"File saved in: '{OUTPUT_JSON}'")

if __name__ == "__main__":
    run_pipeline()