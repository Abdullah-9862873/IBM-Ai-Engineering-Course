# Load Documents from Different Sources

## Course Overview
This course on generative AI applications with RAG and LangChain provides opportunities to apply knowledge to a capstone project on AI applications using retrieval augmented generation (RAG) and LangChain. This course enhances job-ready skills and accelerates careers in AI engineering.

### Prerequisites
- Basic knowledge of Python
- Familiarity with LLMs, LangChain, and RAG (added advantage)

### Learning Outcomes
After completing this course, you will be able to:
1. Use LangChain to load documents from various sources (PDF, CSV, URLs, text)
2. Apply text splitting techniques with RAG and LangChain to enhance model responsiveness
3. Create and configure a vector database to store document embeddings
4. Develop a retriever to fetch document segments based on queries
5. Set up a simple Gradio interface for model interaction
6. Construct a Q&A bot using LangChain and LLM to answer questions from loaded documents

---

## Document Loaders in LangChain

LangChain uses document loaders to gather information from several sources such as websites, files, and databases, and then prepares it for further use. Document loaders act as connectors, pulling in data and converting it into a format LangChain can work with.

### Text Files
For plain text files, use the `TextLoader` class in LangChain:

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("file.txt")
documents = loader.load()
```

Each document object includes `metadata` and `page_content` attributes stored in a list.

### PDF Files
For PDF files, use `PyPDFLoader` or `PyMuPDFLoader`:

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("paper.pdf")
documents = loader.load()  # Array of documents, one per page
```

**PyMuPDFLoader** is the fastest PDF parsing tool in LangChain and includes more comprehensive metadata.

### Markdown Files
Use `UnstructuredMarkdownLoader`:

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("file.md")
documents = loader.load()
```

Note: Content may include many line-break characters.

### JSON Files
Use `JSONLoader` with a JQ schema to parse specific fields:

```python
from langchain.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[].content"
)
documents = loader.load()
```

### CSV Files
Use `CSVLoader` to convert each row into a separate document:

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()  # One document per row
```

For loading all data as one document, use `UnstructuredCSVLoader`.

### Web Pages
For loading web pages, use `WebBaseLoader`:

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader(["https://example.com"])
documents = loader.load()  # Extracts all text, avoids HTML tags
```

For multiple websites, create a list of URLs and pass to the loader.

### Word Documents
Use `Docx2txtLoader`:

```python
from langchain.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
documents = loader.load()
```

### Multiple File Types
For projects with unknown or varied file formats, use `UnstructuredFileLoader`:

```python
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(["file.md", "file.txt"])
documents = loader.load()
```

This loader supports: text files, PowerPoint presentations, HTML pages, PDFs, images, and more.

---

## Best Practices for Loading Documents in LangChain Applications

### 1. Choose the Right Loader Based on Data Source
- **File Loaders**: Use loaders designed for PDFs, CSVs, or text files
- **URL and API Loaders**: Choose loaders that support URLs or REST APIs for online sources

### 2. Optimize Loading Speed
- **Batch Loading**: Process multiple files at once
- **Parallel Processing**: Use `concurrent.futures` or `multiprocessing` for numerous files

### 3. Implement Error Handling for Robustness
- **Retry Mechanism**: Handle intermittent errors (network timeouts)
- **Logging Errors**: Maintain logs for troubleshooting

### 4. Use Caching for Repeated Loads
- **Local Caching**: Save frequently accessed documents locally
- **Set Expiry**: Refresh cached content periodically

### 5. Monitor Resource Usage
- **Memory Management**: Limit documents loaded simultaneously
- **Optimize for Large Files**: Split large documents into smaller chunks before loading

---

## Summary
LangChain provides diverse document loaders for various data sources:
- **Text files**: TextLoader
- **PDF files**: PyPDFLoader, PyMuPDFLoader
- **Markdown**: UnstructuredMarkdownLoader
- **JSON**: JSONLoader
- **CSV**: CSVLoader, UnstructuredCSVLoader
- **Web pages**: WebBaseLoader
- **Word docs**: Docx2txtLoader
- **Multiple formats**: UnstructuredFileLoader

Document loaders are connectors that gather data and convert it into a compatible format for downstream processing in RAG applications.