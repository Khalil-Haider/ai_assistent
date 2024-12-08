import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Union, List

class QwenVLDescriptionGenerator:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-VL-2B-Instruct", 
                 min_pixels: int = 256 * 28 * 28, 
                 max_pixels: int = 720 * 28 * 28,
                 device: str = "auto"):
        """
        Initialize the Qwen-VL Description Generator.
        
        Args:
            model_name (str): Hugging Face model name
            min_pixels (int): Minimum pixel count for image processing
            max_pixels (int): Maximum pixel count for image processing
            device (str): Device to run the model on (auto, cuda, cpu)
        """
        # Load the model and processor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map=device
        )
        
        # Create processor with resolution limits
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
    
    def process_input(self, input_path: str) -> Dict[str, Union[str, Image.Image]]:
        """
        Process an input path. Detects if it's an image or video.
        
        Args:
            input_path (str): Path to the input file
        
        Returns:
            Dict containing input type and data
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file type is unsupported
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file {input_path} does not exist.")
        
        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Supported video extensions
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv')
        
        # Determine file type
        input_path_lower = input_path.lower()
        if input_path_lower.endswith(image_extensions):
            image = Image.open(input_path)
            return {"type": "image", "image": image}
        elif input_path_lower.endswith(video_extensions):
            return {"type": "video", "video": input_path}
        else:
            raise ValueError(f"Unsupported file type: {input_path}")
    
    def generate_description(self, input_path: str, max_new_tokens: int = 128) -> str:
        """
        Generate a description for an image or video file.
        
        Args:
            input_path (str): Path to the input file
            max_new_tokens (int): Maximum number of tokens to generate
        
        Returns:
            str: Generated description
        """
        # Process input file
        input_data = self.process_input(input_path)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user", 
                "content": [input_data, {"type": "text", "text": "Describe this input."}]
            }
        ]
        
        # Preprocess input for the model
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process vision information
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        inputs = inputs.to(self.model.device)
        
        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode and return output
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

