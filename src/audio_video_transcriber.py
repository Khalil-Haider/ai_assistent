import os
import torch
import tempfile
import torchaudio
from typing import Optional
from dataclasses import dataclass
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)
from moviepy.editor import VideoFileClip
import soundfile as sf

@dataclass
class AudioTranscriber:
    model_id: str = "models\distil-whisper-large-v3"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def __post_init__(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=25,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

    def _convert_video_to_audio(self, input_file: str, output_file: str) -> None:
        """Convert video to mono audio."""
        video = VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(
            output_file,
            fps=16000,
            nbytes=2,
            codec='pcm_s16le'
        )
        audio.close()
        video.close()

    def _convert_audio_to_mono(self, input_file: str, output_file: str) -> None:
        """Convert audio to mono WAV."""
        audio, sample_rate = sf.read(input_file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        sf.write(output_file, audio, 16000, subtype='PCM_16')

    def transcribe(self, input_file: str) -> Optional[str]:
        """
        Transcribe audio/video file to text.
        
        Args:
            input_file: Path to input audio/video file
        
        Returns:
            Transcribed text or None if transcription fails
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Prepare mono audio file
                temp_wav_file = os.path.join(temp_dir, "temp_audio.wav")
                
                # Convert input to mono WAV
                if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self._convert_video_to_audio(input_file, temp_wav_file)
                else:
                    self._convert_audio_to_mono(input_file, temp_wav_file)
                
                # Load and process audio
                audio_input, sampling_rate = torchaudio.load(temp_wav_file)

                # Ensure mono channel
                if audio_input.shape[0] > 1:  # Stereo to mono
                    audio_input = torch.mean(audio_input, dim=0, keepdim=True)

                # Resample if necessary
                if sampling_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                    audio_input = resampler(audio_input)

                # Transcribe
                audio_array = audio_input.squeeze().numpy()
                result = self.pipe(audio_array)  # Removed 'return_tensors'
                return result["text"]
            
            except Exception as e:
                print(f"Transcription error: {e}")
                return None













































