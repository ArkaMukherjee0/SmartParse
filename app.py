import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from rouge import Rouge
import os
from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()  # Load environment variables from .env file

key = os.getenv('GROQ_API_KEY')
print(key)
# Function to calculate ROUGE scores
def calculate_rouge_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

# Function to initialize conversation chain with Mixtral 8x7B language model
# We use GroqCloud to efficiently handle inference on any device

llm_groq = ChatGroq(model="mixtral-8x7b-32768",
                    groq_api_key=key)

# llm_groq = AutoModelForCausalLM.from_pretrained("Nexusflow/Athene-70B")

# Streamlit app
st.set_page_config(page_title="SmartParse: Your AI-Powered PDF Companion", layout="centered")

st.markdown(f'<h1 class="title">SmartParse: Your AI-Powered PDF Companion</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Please upload a PDF file to get started!", type="pdf")

st.sidebar.title("SmartParse: Unlock PDF Insights")
st.sidebar.markdown("""
    🌟 **Introducing SmartParse: Instantly Unlock Your PDFs!** 📚

SmartParse instantly transforms complex PDFs into accessible insights, saving you time and effort with every document.

    """)
st.sidebar.markdown("""
                📄[GitHub repo](https://github.com/ArkaMukherjee0/SmartParse)
                """)

st.sidebar.markdown("""
    💡 **How SmartParse Works**
    
    Simply upload your PDF, and let SmartParse leverage Retrieval-Augmented Generation (RAG) to process it. By combining advanced document parsing with intelligent retrieval techniques, SmartParse not only extracts key information but also enables you to ask specific questions about the PDF's content. It’s like having a personal assistant that reads, remembers, and retrieves answers instantly, tailored to your needs.
    """)

st.sidebar.markdown("""
    ### 🖥️ Technologies Used in SmartParse

1. **Streamlit**  
   - A Python library for creating interactive web applications with minimal code. Here, it's used to build the SmartParse UI for uploading and interacting with PDF files.

2. **ChatGroq (with `ChatGroq` model)**  
   - A conversational model setup for SmartParse, providing natural language generation for responding to user queries on PDF content.

3. **PyPDF2**  
   - A Python library for reading and extracting text from PDF files, which enables text processing within SmartParse.

4. **RecursiveCharacterTextSplitter**  
   - Splits long text into manageable chunks, allowing efficient processing and retrieval, especially useful for handling large PDF documents.

5. **FAISS (Facebook AI Similarity Search)**  
   - A library for efficient similarity search and clustering of dense vectors, used here for storing and retrieving vectorized text data.

6. **Hugging Face Embeddings**  
   - The "sentence-transformers/all-MiniLM-L6-v2" model generates sentence embeddings, enabling semantic similarity between user queries and PDF content.

7. **Chroma**  
   - A vector store framework, allowing fast and efficient storage and retrieval of vector embeddings. In this case, it's used to store and retrieve chunks of text from the PDF.

8. **ChatMessageHistory**  
   - A component to store chat message history, preserving context in conversations, especially valuable for continuous queries.

9. **ConversationBufferMemory**  
    - A memory object that retains chat history for ongoing conversation context, essential for generating contextually relevant responses.

10. **ConversationalRetrievalChain**  
    - Combines language models with retrieval mechanisms, allowing users to ask questions based on retrieved document information, providing accurate and context-aware answers.

    """)

# Implementing KeySage
if uploaded_file:
    # Inform the user that processing has started
    with st.spinner(f"Working on `{uploaded_file.name}`..."):
        # Read the PDF file
        pdf = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        # Create a suitable output filename
        txt_filename = os.path.splitext(uploaded_file.name)[0] + ".txt"  
        print(txt_filename)
        # Write the extracted text to the file
        with open(txt_filename, "w", encoding="utf-8") as txt_file:
            txt_file.write(pdf_text)
  
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)

        # Create metadata for each chunk
        # metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Create a FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        docsearch = FAISS.from_texts(texts, embeddings) #metadatas=metadatas)

        # Initialize message history for conversation
        message_history = ChatMessageHistory()

        # Memory for conversational context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create a chain that uses the FAISS vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_groq,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

    st.success(f"Successfully processed `{uploaded_file.name}`. You can now ask questions!")

    #while True:
    user_input = st.text_input("Ask any questions about the PDF:")

    if user_input:
        # Call the chain with user's message content
        res = chain.invoke(user_input)
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []  # Initialize list to store text elements

        # Process source documents if available
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(source_doc.page_content)
            source_names = [f"source_{idx}" for idx in range(len(source_documents))]

            # Add source references to the answer
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

        # Display the results
        st.markdown(f"**Answer:** {answer}")

        for idx, element in enumerate(text_elements):
            with st.expander(f"Source {idx}"):
                st.write(element)

        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_scores(pdf_text, answer)

        # Display ROUGE scores
        st.subheader("ROUGE Scores")
        st.write(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
        st.write(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
        st.write(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
