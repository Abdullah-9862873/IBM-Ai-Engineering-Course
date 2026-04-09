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

---

## Text Splitters in LangChain

After loading documents, the next step is transforming them to suit the application better. You might need to split a document into smaller chunks that can fit into an LLM's context window.

### How Text Splitters Work

1. **Break into small chunks**: First, break the text into small, semantically meaningful chunks (often sentences)
2. **Combine into larger chunks**: Combine these small chunks into larger ones aiming for a specific size
3. **Create new chunk**: Once the size is reached, designate that chunk as separate
4. **Overlap**: Create a new chunk with some overlap from the previous one to maintain context

### Two Axes of Text Splitters

1. **How text is split**: The method/strategy used to break text (characters, words, sentences, or custom tokens)
2. **How chunk size is measured**: Criteria for when a chunk is complete (characters, words, tokens, or custom metrics)

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Separator** | Character(s) used to split text (line change, paragraph, space) | By paragraphs |
| **Chunk Size** | Maximum number of characters each chunk can contain | 1,000 |
| **Chunk Overlap** | Number of characters that overlap between consecutive chunks | 200 |
| **Length Function** | How the length of chunks is calculated | - |

---

## Commonly Used Text Splitters

### 1. Character Text Splitter
The simplest method - splits based on characters/separators:

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_text(text)
```

- Splits text by customized separator
- Can set overlaps between chunks to ensure information isn't lost

### 2. Recursive Character Text Splitter
Uses recursion to split text - recommended for generic text:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=0
)
chunks = splitter.split_text(text)
```

How it works:
- Takes large text and tries to split it up until chunks are small enough
- Uses a set of characters: by paragraphs, sentences, words, or characters
- Tries first separator (paragraphs), then assesses each chunk
- If chunk exceeds size, moves to next level (sentences), and so on

### 3. Code Text Splitter
Splits code with multiple language support:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_code(code)
```

Supported languages include: Python, JavaScript, C, C++, Go, Java, PHP, Ruby, Swift, and more.

### 4. Markdown Header Text Splitter
Splits markdown files by specified headers to honor document structure:

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = splitter.split_text(markdown_text)
```

This splits markdown content within each header group, maintaining logical structure.

---

## Summary

LangChain text splitters allow you to:
- Split long documents into smaller chunks that fit into LLM context windows
- Maintain semantic meaning and context between chunks
- Use various splitting strategies (characters, sentences, paragraphs, headers)
- Control chunk size and overlap for optimal processing

Key splitters:
- **CharacterTextSplitter**: Simple character-based splitting
- **RecursiveCharacterTextSplitter**: Best for generic text, uses multiple separators
- **CodeTextSplitter**: For code with language-specific splitting
- **MarkdownHeaderTextSplitter**: For markdown files, respects header structure