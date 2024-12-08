import streamlit as st
import os
import uuid
from typing import Optional

# Import the custom classes
from audio_video_transcriber import AudioTranscriber
from pdf_conversational_rag_chatbot import PDFChatbot
from qwen_vl_description_generator import QwenVLDescriptionGenerator

class MultimodalAIAssistant:
    def __init__(self):
        """
        Initialize the Multimodal AI Assistant with session state management
        """
        # Initialize session state for tracking active sessions
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize models as None, will be loaded when needed
        self.audio_transcriber = None
        self.pdf_chatbot = None
        self.image_description_generator = None
        
        # Google API key management
        if 'google_api_key' not in st.session_state:
            st.session_state.google_api_key = None

    def load_audio_transcriber(self):
        """
        Load Audio Transcriber model
        """
        if not self.audio_transcriber:
            try:
                self.audio_transcriber = AudioTranscriber()
                st.success("Audio Transcriber initialized successfully!")
            except Exception as e:
                st.error(f"Failed to load Audio Transcriber: {e}")

    def load_pdf_chatbot(self, pdf_path):
        """
        Load PDF Chatbot with the given PDF
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        # Ensure Google API key is set
        if not st.session_state.google_api_key:
            st.warning("Please enter Google API key first!")
            return None

        try:
            self.pdf_chatbot = PDFChatbot(
                pdf_path=pdf_path, 
                google_api_key=st.session_state.google_api_key
            )
            st.success("PDF Chatbot initialized successfully!")
            return self.pdf_chatbot
        except Exception as e:
            st.error(f"Failed to load PDF Chatbot: {e}")
            return None

    def load_image_description_generator(self):
        """
        Load Qwen VL Description Generator
        """
        if not self.image_description_generator:
            try:
                self.image_description_generator = QwenVLDescriptionGenerator()
                st.success("Image Description Generator initialized successfully!")
            except Exception as e:
                st.error(f"Failed to load Image Description Generator: {e}")

    def render_sidebar(self):
        """
        Render the sidebar with configuration options
        """
        st.sidebar.title("Multimodal AI Assistant")
        
        # Google API Key Input
        st.sidebar.subheader("Google API Configuration")
        google_api_key = st.sidebar.text_input(
            "Enter Google API Key", 
            type="password", 
            value=st.session_state.google_api_key or ""
        )
        if google_api_key:
            st.session_state.google_api_key = google_api_key
        
        # Clear History Button
        if st.sidebar.button("Clear Conversation History"):
            if self.pdf_chatbot:
                self.pdf_chatbot.clear_history(st.session_state.session_id)
            st.success("Conversation history cleared!")

    def main_ui(self):
        """
        Render the main user interface
        """
        st.title("Multimodal AI Assistant")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload File", 
            type=['pdf', 'txt', 'mp3', 'wav', 'mp4', 'avi', 'jpg', 'jpeg', 'png']
        )
        
        # Processing mode selection
        processing_mode = st.selectbox(
            "Select Processing Mode", 
            [
                "Auto Detect", 
                "Question Answering", 
                "Transcription", 
                "Image Description"
            ]
        )
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())
            file_path = os.path.join("temp", uploaded_file.name)
            
            # Determine file type and processing mode
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Auto-detect processing based on file type and selected mode
            if processing_mode == "Auto Detect":
                if file_extension in ['.pdf']:
                    processing_mode = "Question Answering"
                elif file_extension in ['.jpg', '.jpeg', '.png']:
                    processing_mode = "Image Description"
                elif file_extension in ['.mp3', '.wav', '.mp4', '.avi']:
                    processing_mode = "Transcription"
            
            # Process based on mode
            if processing_mode == "Question Answering" and file_extension == '.pdf':
                self.load_pdf_chatbot(file_path)
                if self.pdf_chatbot:
                    query = st.text_input("Ask a question about the PDF")
                    if query:
                        result = self.pdf_chatbot.chat(query, st.session_state.session_id)
                        st.write(result['answer'])
            
            elif processing_mode == "Transcription" and file_extension in ['.mp3', '.wav', '.mp4', '.avi']:
                self.load_audio_transcriber()
                if self.audio_transcriber:
                    transcription = self.audio_transcriber.transcribe(file_path)
                    st.subheader("Transcription")
                    st.write(transcription)
            
            elif processing_mode == "Image Description" and file_extension in ['.jpg', '.jpeg', '.png']:
                self.load_image_description_generator()
                if self.image_description_generator:
                    description = self.image_description_generator.generate_description(file_path)
                    st.subheader("Image Description")
                    st.write(description)
    
    def run(self):
        """
        Run the Streamlit application
        """
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Render sidebar and main UI
        self.render_sidebar()
        self.main_ui()

def main():
    assistant = MultimodalAIAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
