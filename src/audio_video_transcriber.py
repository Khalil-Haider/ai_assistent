import os
import subprocess
import tempfile
import torch
import logging
from typing import Optional, Union

import torchaudio
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)

class AudioTranscriber:
    """
    A comprehensive audio transcription class using Distil-Whisper model
    
    Supports multiple audio and video file formats with robust error handling
    """
    
    def __init__(
        self, 
        model_id: str = "distil-whisper/distil-large-v3",
        chunk_length_s: int = 25,
        max_new_tokens: int = 128,
        batch_size: int = 16,
        log_level: int = logging.INFO
    ):
        """
        Initialize the AudioTranscriber
        
        Parameters:
        -----------
        model_id : str, optional
            Whisper model to use (default: distil-whisper/distil-large-v3)
        chunk_length_s : int, optional
            Length of audio chunks to process (default: 25 seconds)
        max_new_tokens : int, optional
            Maximum number of new tokens for transcription (default: 128)
        batch_size : int, optional
            Batch processing size (default: 16)
        log_level : int, optional
            Logging level (default: logging.INFO)
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Device and precision configuration
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.logger.info(f"Using device: {self.device}")
        
        try:
            # Load Whisper model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.model.to(self.device)
            
            # Load processor
            self.processor = AutoProcessor.from_pretained(model_id)
            
            # Create transcription pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            self.logger.info("Transcription model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize transcription model: {e}")
            raise
    
    def _convert_to_wav(
        self, 
        input_file: str, 
        temp_dir: tempfile.TemporaryDirectory
    ) -> str:
        """
        Convert input audio/video file to mono WAV
        
        Parameters:
        -----------
        input_file : str
            Path to input audio/video file
        temp_dir : tempfile.TemporaryDirectory
            Temporary directory for conversion
        
        Returns:
        --------
        str: Path to converted WAV file
        """
        temp_wav_file = os.path.join(temp_dir.name, "temp_audio.wav")
        
        try:
            subprocess.run([
                'ffmpeg', 
                '-i', input_file, 
                '-ac', '1',  # Convert to mono
                '-ar', '16000',  # Resample to 16kHz
                temp_wav_file
            ], check=True, capture_output=True)
            
            self.logger.info(f"Successfully converted {input_file} to {temp_wav_file}")
            return temp_wav_file
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg conversion error: {e.stderr.decode()}")
            raise RuntimeError(f"Audio conversion failed: {e.stderr.decode()}")
    
    def transcribe(
        self, 
        input_file: str, 
        output_file: Optional[str] = None
    ) -> str:
        """
        Transcribe an audio or video file
        
        Parameters:
        -----------
        input_file : str
            Path to input audio/video file
        output_file : str, optional
            Path to save transcription text file
        
        Returns:
        --------
        str: Transcription text
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.TemporaryDirectory()
            
            # Convert to WAV
            wav_file = self._convert_to_wav(input_file, temp_dir)
            
            # Load and process audio
            audio_input, sampling_rate = torchaudio.load(wav_file)
            
            # Transcribe
            result = self.pipe(audio_input.squeeze().numpy())
            transcription = result["text"]
            
            # Optional: Write to output file
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                self.logger.info(f"Transcription saved to {output_file}")
            
            return transcription
        
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        
        finally:
            # Cleanup temporary files
            if temp_dir:
                temp_dir.cleanup()
                self.logger.info("Temporary files cleaned up")
    
    def batch_transcribe(
        self, 
        input_files: Union[str, list], 
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Transcribe multiple audio/video files
        
        Parameters:
        -----------
        input_files : str or list
            Single file path or list of file paths
        output_dir : str, optional
            Directory to save transcription files
        
        Returns:
        --------
        dict: Mapping of input files to transcriptions
        """
        # Handle single file input
        if isinstance(input_files, str):
            input_files = [input_files]
        
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Transcribe files
        transcriptions = {}
        for file in input_files:
            try:
                output_file = None
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    output_file = os.path.join(output_dir, f"{base_name}_transcription.txt")
                
                transcription = self.transcribe(file, output_file)
                transcriptions[file] = transcription
            
            except Exception as e:
                self.logger.error(f"Failed to transcribe {file}: {e}")
                transcriptions[file] = None
        
        return transcriptions

