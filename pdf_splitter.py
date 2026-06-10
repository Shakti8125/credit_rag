import os
import glob
from pypdf import PdfReader, PdfWriter

def split_large_pdf(input_file_path, output_folder, chunk_size=50):
    """Splits a single large PDF into smaller chunks."""
    base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    print(f"\n📖 Processing: {base_filename}.pdf")
    
    try:
        reader = PdfReader(input_file_path)
        total_pages = len(reader.pages)
        print(f"   Total pages: {total_pages}")
        
        # If the document is already smaller than the chunk size, just copy it over as Part 1
        if total_pages <= chunk_size:
            print(f"   Document is under {chunk_size} pages. Saving as a single part.")
            
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)
            part_number = (start_page // chunk_size) + 1
            
            writer = PdfWriter()
            
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
                
            output_filename = f"{base_filename}_Part{part_number}.pdf"
            output_filepath = os.path.join(output_folder, output_filename)
            
            with open(output_filepath, "wb") as output_file:
                writer.write(output_file)
                
            print(f"   ✅ Saved: {output_filename} (Pages {start_page + 1} to {end_page})")
            
    except Exception as e:
        print(f"   ❌ Error processing {base_filename}: {e}")

def batch_process_directory():
    """Loops through all PDFs in the raw folder and splits them into the base folder."""
    # --- CONFIGURATION ---
    RAW_DIR = r"C:\Users\Shakti\Desktop\credit-risk-rag\ingestion\data\raw"      # Where you put the massive PDFs
    OUTPUT_DIR =r"C:\Users\Shakti\Desktop\credit-risk-rag\ingestion\data\splitted_docs"    # Where Docling will look for them
    PAGES_PER_SPLIT = 30              # Safe memory limit for 16GB RAM
    
    # Ensure both directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all PDFs in the raw directory
    pdf_files = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in '{RAW_DIR}'. Please add your files and run again.")
        return

    print(f"🚀 Found {len(pdf_files)} PDFs to process. Starting batch split...")
    
    # The Loop!
    for pdf_path in pdf_files:
        split_large_pdf(pdf_path, OUTPUT_DIR, PAGES_PER_SPLIT)
        
    print(f"\n🎉 Batch splitting complete! All sliced files are ready in '{OUTPUT_DIR}'.")
    print("You can now run: python ingest_documents.py")

if __name__ == "__main__":
    batch_process_directory()