"""
Multi-modal LLM interface for visual knowledge extraction.

Supports multiple backends:
- OpenAI API (GPT-4o, GPT-4o-mini)
- Local models (Qwen-VL, LLaVA, etc.)
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import time


class MLLMInterface:
    """Base interface for multi-modal LLMs."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize MLLM interface.

        Args:
            model_name: Model identifier
            **kwargs: Backend-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs

    def extract_from_image(
        self,
        image: Union[Image.Image, str, Path],
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Extract knowledge from a single image.

        Args:
            image: PIL Image object or path to image file
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Model output text

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement extract_from_image")


class OpenAIMLLM(MLLMInterface):
    """OpenAI API interface for GPT-4o/GPT-4o-mini."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI MLLM.

        Args:
            model_name: OpenAI model name (gpt-4o, gpt-4o-mini)
            api_key: OpenAI API key (if None, read from env)
            base_url: Optional base URL for OpenAI-compatible APIs (e.g., local vLLM)
            **kwargs: Additional OpenAI parameters
        """
        super().__init__(model_name, **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not found. Install with: pip install openai"
            )

        # Initialize client with optional base_url
        client_kwargs = {}

        # API key handling
        if api_key is not None:
            client_kwargs['api_key'] = api_key
        # else: will try to use OPENAI_API_KEY from environment

        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)

    def _encode_image(self, image: Union[Image.Image, str, Path]) -> str:
        """
        Encode image to base64 string.

        Args:
            image: PIL Image or path to image

        Returns:
            Base64 encoded image string
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Encode to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def extract_from_image(
        self,
        image: Union[Image.Image, str, Path],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Extract knowledge using OpenAI Vision API.

        Args:
            image: PIL Image object or path to image file
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional OpenAI parameters

        Returns:
            Model output text
        """
        # Encode image
        image_b64 = self._encode_image(image)

        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]

        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return response.choices[0].message.content


class LocalMLLM(MLLMInterface):
    """Interface for local multi-modal models (Qwen-VL, LLaVA, etc.)."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize local MLLM.

        Args:
            model_name: Model identifier (e.g., "qwen-vl", "llava")
            model_path: Path to model checkpoint
            device: Device to run model on
            **kwargs: Model-specific parameters
        """
        super().__init__(model_name, **kwargs)
        self.device = device
        self.model_path = model_path

        # Model will be loaded lazily
        self.model = None
        self.processor = None

    def _load_model(self):
        """Load model and processor (lazy initialization)."""
        if self.model is not None:
            return

        if "qwen" in self.model_name.lower():
            self._load_qwen()
        elif "llava" in self.model_name.lower():
            self._load_llava()
        else:
            raise ValueError(f"Unsupported local model: {self.model_name}")

    def _load_qwen(self):
        """Load Qwen-VL model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Transformers library not found. "
                "Install with: pip install transformers"
            )

        print(f"Loading Qwen-VL from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True
        )
        self.processor = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        print("Model loaded successfully!")

    def _load_llava(self):
        """Load LLaVA model."""
        raise NotImplementedError("LLaVA support coming soon")

    def extract_from_image(
        self,
        image: Union[Image.Image, str, Path],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Extract knowledge using local model.

        Args:
            image: PIL Image object or path to image file
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Model-specific parameters

        Returns:
            Model output text
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Model-specific inference
        if "qwen" in self.model_name.lower():
            return self._infer_qwen(
                image, system_prompt, user_prompt,
                temperature, max_tokens, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Inference not implemented for {self.model_name}"
            )

    def _infer_qwen(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Run inference with Qwen-VL."""
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # This is a placeholder - actual implementation depends on Qwen-VL API
        # You'll need to adapt this based on the specific model version
        raise NotImplementedError(
            "Qwen-VL inference needs to be implemented based on your model version"
        )


def create_mllm(
    backend: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> MLLMInterface:
    """
    Factory function to create MLLM interface.

    Args:
        backend: Backend type ("openai" or "local")
        model_name: Model name (backend-specific)
        **kwargs: Additional parameters for backend

    Returns:
        MLLMInterface instance

    Raises:
        ValueError: If backend not supported
    """
    if backend == "openai":
        model_name = model_name or "gpt-4o-mini"
        return OpenAIMLLM(model_name=model_name, **kwargs)
    elif backend == "local":
        if not model_name:
            raise ValueError("model_name required for local backend")
        return LocalMLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# Example usage
if __name__ == '__main__':
    import argparse
    from prompts import PromptTemplates

    parser = argparse.ArgumentParser(description='Test MLLM interface')
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'local'],
                        help='MLLM backend')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (if using OpenAI)')

    args = parser.parse_args()

    # Create MLLM
    print(f"Creating {args.backend} MLLM...")
    mllm = create_mllm(
        backend=args.backend,
        model_name=args.model,
        api_key=args.api_key
    )

    # Get prompts
    system_prompt = PromptTemplates.get_phase1_system_prompt()
    user_prompt = PromptTemplates.get_phase1_user_prompt("Test Movie")

    # Extract knowledge
    print(f"\nExtracting knowledge from: {args.image}")
    start_time = time.time()

    output = mllm.extract_from_image(
        image=args.image,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1000
    )

    elapsed = time.time() - start_time

    print(f"\nExtraction completed in {elapsed:.2f}s")
    print("\n" + "="*60)
    print("Model Output:")
    print("="*60)
    print(output)

    # Parse output
    knowledge_points = PromptTemplates.parse_extraction_output(output)
    print("\n" + "="*60)
    print(f"Parsed {len(knowledge_points)} knowledge points:")
    print("="*60)
    print(PromptTemplates.format_knowledge_points(knowledge_points, 'markdown'))
