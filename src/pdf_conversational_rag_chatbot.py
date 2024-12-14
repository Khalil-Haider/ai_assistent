import os
import uuid
import pymupdf4llm
from typing import List, Dict, Optional, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class PDFChatbot:
    """
    A comprehensive PDF Chatbot using Retrieval-Augmented Generation (RAG)
    
    Supports:
    - PDF text extraction
    - Semantic search
    - Contextual question answering
    - Conversation history management
    """
    
    def __init__(
        self, 
        pdf_path: str, 
        google_api_key: str, 
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 600, 
        chunk_overlap: int = 90,
        temperature: float = 0.0
    ):
        """
        Initialize the PDF Chatbot
        
        Parameters:
        -----------
        pdf_path : str
            Path to the PDF file
        google_api_key : str
            Google API key for Gemini model
        embedding_model : str, optional
            Embedding model to use (default: BAAI/bge-m3)
        chunk_size : int, optional
            Size of text chunks (default: 600)
        chunk_overlap : int, optional
            Overlap between text chunks (default: 90)
        temperature : float, optional
            LLM temperature setting (default: 0.0)
        """
        # Set environment variable for Google API
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(pdf_path)
        pages = [Document(page_content=md_text, metadata={"source": pdf_path})]
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)
        
        # Embeddings
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model)
        chunks = [doc.page_content for doc in texts]
        embeddings = self.embedding_model.embed_documents(chunks)
        
        # Vector store
        text_embedding_pairs = list(zip(chunks, embeddings))
        self.vectorstore = FAISS.from_embeddings(text_embedding_pairs, self.embedding_model)
        
        # Language Model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Retrievers
        bm25_retriever = BM25Retriever.from_documents(texts)
        bm25_retriever.k = 9
        faiss_retriever = self.vectorstore.as_retriever()
        
        # Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        # Reranker
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, compression_retriever, contextualize_q_prompt
        )
        
        # QA system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Chains
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Chat history management
        self.store: Dict[str, ChatMessageHistory] = {}
        
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]
        
        # Final conversational RAG chain
        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        # Generate a default session ID
        self.default_session_id = str(uuid.uuid4())
    
    def chat(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a query to the chatbot and get a response
        
        Parameters:
        -----------
        query : str
            User's input query
        session_id : str, optional
            Specific session ID to maintain conversation context
        
        Returns:
        --------
        Dict containing the full response
        """
        # Use default session ID if not provided
        current_session_id = session_id or self.default_session_id
        
        # Invoke the chain
        result = self.conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": current_session_id}},
        )
        
        return result
    
    def clear_history(self, session_id: Optional[str] = None) -> None:
        """
        Clear conversation history for a specific or default session
        
        Parameters:
        -----------
        session_id : str, optional
            Session ID to clear. If None, clears default session.
        """
        current_session_id = session_id or self.default_session_id
        
        if current_session_id in self.store:
            self.store[current_session_id] = ChatMessageHistory()
        
    def list_sessions(self) -> List[str]:
        """
        List all active session IDs
        
        Returns:
        --------
        List of session IDs
        """
        return list(self.store.keys())
    