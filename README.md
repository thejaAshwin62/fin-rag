# fin-rag

fin-rag is a Python-based project designed to [briefly describe the main purpose or goal of your project, e.g., "analyze financial data using Retrieval-Augmented Generation (RAG) techniques"]. This repository is 100% Python and aims to provide efficient solutions for [your project's focus: e.g., data processing, analysis, APIs, etc.].

## Features

- [Feature 1: e.g., "Efficient financial data retrieval and analysis"]
- [Feature 2: e.g., "Supports integration with various data sources"]
- [Feature 3: e.g., "Customizable analysis through RAG pipelines"]
- [Add more features as appropriate]

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/thejaAshwin62/fin-rag.git
   cd fin-rag
   ```

2. **(Optional) Set up a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Linux/macOS
   venv\Scripts\activate     # on Windows
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Required:
   - `GOOGLE_API_KEY`

   Optional ChromaDB settings:
   - `CHROMA_PERSIST_DIR` (default: `./chroma_db`)
   - `CHROMA_COLLECTION_NAME` (default: `finance-rag-chatbot`)

## Usage

[Provide example code or CLI commands for how to use the project.]

```python
from fin_rag import main

result = main.run_analysis("input_data.csv")
print(result)
```

Or, from the command line:

```bash
python main.py
```

This project now uses **ChromaDB** as the vector store for ingestion and semantic search.

## Project Structure

```
fin-rag/
├── fin_rag/
│   └── ...                # Source code files
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests to discuss improvements or report bugs.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

[MIT](LICENSE) (or specify your license here).

## Contact

Created by [thejaAshwin62](https://github.com/thejaAshwin62) – feel free to reach out!

---

> _This project is 100% Python._
