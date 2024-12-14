import os
import uuid
import streamlit as st
import tempfile
import torch

# Import custom classes
from whisper_transcription_app import AudioTranscriber
from pdf_conversational_rag_chatbot import PDFChatbot

import os
import uuid
import streamlit as st
import tempfile
import torch

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Transcription & Q&A Assistant",
        page_icon="🤖",
        layout="wide"
    )

def initialize_session_state():
    """Initialize or reset session state variables"""
    if 'transcription_text' not in st.session_state:
        st.session_state.transcription_text = None
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'show_transcription_text' not in st.session_state:
        st.session_state.show_transcription_text = False
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
    if 'pdf_uploaded_directly' not in st.session_state:
        st.session_state.pdf_uploaded_directly = False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name

def create_pdf_from_text(text, prefix='transcription'):
    """Create a temporary PDF from transcribed text"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', prefix=prefix) as temp_pdf:
        temp_pdf_path = temp_pdf.name
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Register a TrueType font
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))

    c = canvas.Canvas(temp_pdf_path, pagesize=letter)
    width, height = letter
    c.setFont('Arial', 10)
    
    # Wrap text
    text_lines = []
    words = text.split()
    current_line = []
    current_line_length = 0

    for word in words:
        if current_line_length + len(word) > 80:
            text_lines.append(' '.join(current_line))
            current_line = [word]
            current_line_length = len(word)
        else:
            current_line.append(word)
            current_line_length += len(word) + 1

    text_lines.append(' '.join(current_line))

    # Write lines
    y = height - inch
    for line in text_lines:
        c.drawString(inch, y, line)
        y -= 12
        if y <= inch:
            c.showPage()
            c.setFont('Arial', 10)
            y = height - inch

    c.save()
    return temp_pdf_path

def main():
    setup_page_config()
    initialize_session_state()

    st.title("Transcription & Q&A Assistant")

    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload Audio/Video/PDF File", 
        type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'pdf']
    )

    # Mode and Text Visibility Section
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Select Mode", 
            ["Transcription", "Question Answering"], 
            horizontal=True
        )
    
    with col2:
        # Toggle for showing transcription text
        if st.session_state.transcription_text:
            st.session_state.show_transcription_text = st.checkbox(
                "Show Transcription Text", 
                value=st.session_state.show_transcription_text
            )

    # Google API Key Input
    google_api_key = st.text_input(
        "Enter Google API Key", 
        type="password",
        help="Required for RAG functionality using Gemini"
    )

    if uploaded_file and google_api_key:
        # Determine file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file
        temp_file_path = save_uploaded_file(uploaded_file)
        st.session_state.file_type = file_extension

        # PDF Upload Handling
        if file_extension == '.pdf':
            st.session_state.pdf_path = temp_file_path
            st.session_state.transcription_text = None  # Reset transcription text
            st.session_state.pdf_uploaded_directly = True
            
            # Automatically initialize RAG pipeline for PDF only if not already initialized
            if st.session_state.chatbot is None:
                with st.spinner("Initializing RAG Pipeline for PDF..."):
                    st.session_state.chatbot = PDFChatbot(
                        pdf_path=st.session_state.pdf_path,
                        google_api_key=google_api_key
                    )
                st.success("PDF Loaded and RAG Pipeline Initialized!")

        # Audio/Video Transcription
        elif file_extension in ['.wav', '.mp3', '.mp4', '.avi', '.mov']:
            # Reset PDF direct upload flag
            st.session_state.pdf_uploaded_directly = False
            
            # Transcription Mode
            if mode == "Transcription":
                with st.spinner("Transcribing..."):
                    transcriber = AudioTranscriber()
                    transcription_result = transcriber.transcribe(temp_file_path)
                    
                    if transcription_result:
                        st.session_state.transcription_text = transcription_result
                        st.session_state.pdf_path = create_pdf_from_text(transcription_result)
                        
                        st.success("Transcription Complete!")
                        st.session_state.show_transcription_text = True

        # Q&A Mode
        if mode == "Question Answering":
            # Ensure PDF is loaded for Q&A
            if st.session_state.pdf_path is None:
                st.warning("Please upload a PDF or transcribe an audio/video file first.")

            # Initialize RAG pipeline only for transcribed files, not for directly uploaded PDFs
            if (st.session_state.pdf_path and 
                st.session_state.chatbot is None and 
                not st.session_state.pdf_uploaded_directly):
                with st.spinner("Initializing RAG Pipeline..."):
                    st.session_state.chatbot = PDFChatbot(
                        pdf_path=st.session_state.pdf_path,
                        google_api_key=google_api_key
                    )

        # Transcription Text Display (Always Accessible)
        if st.session_state.transcription_text and st.session_state.show_transcription_text:
            st.subheader("Transcribed Text")
            st.text_area(
                "Transcription", 
                value=st.session_state.transcription_text, 
                height=300, 
                disabled=True
            )

        # Chat Interface for Q&A Mode
        if mode == "Question Answering" and st.session_state.chatbot:
            st.header("Ask Questions about the Document")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("What would you like to know?"):
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get chatbot response
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = st.session_state.chatbot.chat(
                            prompt, 
                            session_id=st.session_state.session_id
                        )
                        st.markdown(response['answer'])

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer']
                })

    # Cleanup and reset
    st.sidebar.header("Session Management")
    if st.sidebar.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()