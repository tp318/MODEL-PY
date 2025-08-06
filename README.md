# RAG (Retrieval-Augmented Generation) API

A FastAPI-based service that implements a Retrieval-Augmented Generation pipeline for document processing and question-answering.

## Features

- Document ingestion from URLs (PDF, DOCX, TXT)
- Document parsing and chunking
- Vector embeddings generation
- Semantic search capabilities
- Question-answering with context
- Secure API with Bearer token authentication

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd WORKING-RAG
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root (optional for development):
   ```
   API_KEY=your_api_key_here
   ```

## Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

### POST /api/v1/hackrx/run

Process a document and answer questions about its content.

**Headers:**
```
Authorization: Bearer c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301
Content-Type: application/json
```

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic of this document?",
        "What are the key points mentioned?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "The document discusses...",
        "The key points are..."
    ]
}
```

## Development

### Project Structure

```
WORKING-RAG/
├── api/
│   └── main.py           # FastAPI application
├── EMBEDDING/           # Embedding and vector storage
├── INGESTION/           # Document download and processing
├── LLM/                 # Language model integration
├── SEARCHING/           # Search functionality
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Testing

To run tests:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
