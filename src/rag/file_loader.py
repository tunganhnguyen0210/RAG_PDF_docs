from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_non_utf8_characters(text: str) -> str:
    """Loại bỏ ký tự không phải UTF-8 (ASCII giữ lại)."""
    return ''.join(char for char in text if ord(char) < 128)


def load_pdf(pdf_file: str):
    """Load PDF thành documents, loại bỏ ký tự lỗi."""
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs


def get_num_cpu():
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        raise NotImplementedError


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs.get("workers", 1))
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded


class TextSplitter:
    def __init__(self, 
                 separators: List[str] = ['\n\n', '\n', ' ', ''],
                 chunk_size: int = 300,
                 chunk_overlap: int = 0) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self, 
                 file_type: str = Literal["pdf"],
                 split_kwargs: dict = {
                     "chunk_size": 300,
                     "chunk_overlap": 0}) -> None:
        
        assert file_type == "pdf", "Only pdf is supported"
        self.file_type = file_type
        self.doc_loader = PDFLoader()
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        return self.doc_splitter(doc_loaded)

    def load_dir(self, dir_path: str, workers: int = 1):
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        assert len(pdf_files) > 0, f"No pdf files found in {dir_path}"
        return self.load(pdf_files, workers=workers)
