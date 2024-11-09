# 📖 SmartParse – Instantly Unlock Your PDFs with RAG!

SmartParse is a powerful PDF processing app leveraging Retrieval-Augmented Generation (RAG) to turn complex documents into accessible insights. Just upload your PDF, ask questions, and receive instant answers, complete with source references. Perfect for anyone who needs to dive into PDF content quickly and efficiently!

---

## 🌟 Features
- **Instant PDF Parsing** – Quickly extract and parse the content of any PDF.
- **Question-Answering** – Ask questions about the document and get direct answers.
- **Memory-Powered Conversations** – Track your chat history for context-aware responses.
- **Efficient Document Retrieval** – Rely on vector-based document retrieval with FAISS for rapid response.
- **Simple, Intuitive Interface** – Built using Streamlit for a clean, easy-to-use experience.

---

## 🖼️ Screenshot
> _Placeholder for screenshot – upload a screenshot of the SmartParse interface here._

---

## 🛠️ Technologies Used

1. **dotenv** – Loads environment variables from a `.env` file, simplifying access to API keys and other secrets.

2. **Streamlit** – A Python library for creating interactive web applications with minimal code. Here, it’s used to build the SmartParse UI for uploading and interacting with PDF files.

3. **ChatGroq (with `ChatGroq` model)** – A conversational model setup for SmartParse, providing natural language generation for responding to user queries on PDF content.

4. **PyPDF2** – A Python library for reading and extracting text from PDF files, which enables text processing within SmartParse.

5. **RecursiveCharacterTextSplitter** – Splits long text into manageable chunks, allowing efficient processing and retrieval, especially useful for handling large PDF documents.

6. **FAISS (Facebook AI Similarity Search)** – A library for efficient similarity search and clustering of dense vectors, used here for storing and retrieving vectorized text data.

7. **Hugging Face Embeddings** – The "sentence-transformers/all-MiniLM-L6-v2" model generates sentence embeddings, enabling semantic similarity between user queries and PDF content.

8. **Chroma** – A vector store framework, allowing fast and efficient storage and retrieval of vector embeddings. In this case, it’s used to store and retrieve chunks of text from the PDF.

9. **ChatMessageHistory** – A component to store chat message history, preserving context in conversations, especially valuable for continuous queries.

10. **ConversationBufferMemory** – A memory object that retains chat history for ongoing conversation context, essential for generating contextually relevant responses.

11. **ConversationalRetrievalChain** – Combines language models with retrieval mechanisms, allowing users to ask questions based on retrieved document information, providing accurate and context-aware answers.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- API Key for GROQ Language Model

### Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/SmartParse.git
    cd SmartParse
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**
    - Create a `.env` file in the project root and add your `GROQ_API_KEY`:
      ```dotenv
      GROQ_API_KEY=your_groq_api_key
      ```

### Run the Application
To start the Streamlit app, run:
```bash
streamlit run app.py

