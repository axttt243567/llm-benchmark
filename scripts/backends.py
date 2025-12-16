#!/usr/bin/env python3
"""
Inference Backends for LLM Benchmark
====================================
Pluggable backend interface for different LLM inference providers.

Backends:
  - OllamaBackend: Local Ollama server (default)
  - TransformersBackend: HuggingFace Transformers (for Colab T4 GPU)
  - RemoteOllamaBackend: Remote Ollama via API (optional)
"""

import subprocess
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

# Check if running in Colab
IS_COLAB = 'google.colab' in str(os.environ.get('COLAB_RELEASE_TAG', ''))


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of available model names."""
        pass
    
    @abstractmethod
    def generate(self, model_name: str, prompt: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            model_name: Name/identifier of the model
            prompt: Input prompt text
            timeout: Maximum time in seconds
            
        Returns:
            Dict with keys:
                - response: str (the generated text)
                - success: bool
                - error_type: Optional[str]
                - duration: float (time taken)
        """
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, str]:
        """Return backend metadata for logging."""
        pass


class OllamaBackend(InferenceBackend):
    """
    Backend for local Ollama server.
    Requires Ollama to be installed and running.
    """
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
    
    def get_available_models(self) -> List[str]:
        """Get models from 'ollama list' command."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            
            models = []
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return []
    
    def generate(self, model_name: str, prompt: str, timeout: int = 300) -> Dict[str, Any]:
        """Generate using Ollama subprocess."""
        import time
        
        start_time = time.time()
        response_text = ""
        success = True
        error_type = None
        
        try:
            process = subprocess.run(
                ['ollama', 'run', model_name, prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )
            
            if process.returncode == 0:
                response_text = process.stdout.strip()
            else:
                response_text = f"ERROR: {process.stderr.strip()}"
                success = False
                error_type = "OLLAMA_ERROR"
                
        except subprocess.TimeoutExpired:
            response_text = f"ERROR: Response timeout ({timeout}s)"
            success = False
            error_type = "TIMEOUT"
        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            success = False
            error_type = "BACKEND_ERROR"
        
        duration = time.time() - start_time
        
        return {
            "response": response_text,
            "success": success,
            "error_type": error_type,
            "duration": round(duration, 3)
        }
    
    def get_backend_info(self) -> Dict[str, str]:
        return {
            "backend": "ollama",
            "host": self.host,
            "port": str(self.port)
        }


class TransformersBackend(InferenceBackend):
    """
    Backend for HuggingFace Transformers.
    Optimized for Google Colab T4 GPU (16GB VRAM).
    
    Supports:
      - FP16 inference for efficiency
      - 4-bit quantization for larger models
      - Automatic device placement
    """
    
    def __init__(
        self,
        device: str = "auto",
        torch_dtype: str = "float16",
        use_4bit: bool = False,
        max_new_tokens: int = 512,
        models_to_load: Optional[List[str]] = None
    ):
        """
        Initialize TransformersBackend.
        
        Args:
            device: "cuda", "cpu", or "auto"
            torch_dtype: "float16", "bfloat16", or "float32"
            use_4bit: Enable 4-bit quantization (requires bitsandbytes)
            max_new_tokens: Maximum tokens to generate
            models_to_load: Optional list of model paths/names to preload
        """
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        
        # Lazy imports - only import when needed
        self._torch = None
        self._transformers = None
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_tokenizers: Dict[str, Any] = {}
        
        # Common model name mappings (Ollama name -> HuggingFace name)
        self.model_mappings = {
            # Llama models
            "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
            # Gemma models
            "gemma3:1b": "google/gemma-3-1b-it",
            "gemma3:4b": "google/gemma-3-4b-it",
            "gemma2:2b": "google/gemma-2-2b-it",
            # Qwen models
            "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen3:8b": "Qwen/Qwen3-8B",
            # Phi models
            "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
            "phi3.5:3.8b": "microsoft/Phi-3.5-mini-instruct",
            # Mistral models
            "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
            # SmolLM models
            "smollm2:135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "smollm2:360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "smollm2:1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        }
    
    def _ensure_imports(self):
        """Lazy import torch and transformers."""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                
                # Set dtype
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32
                }
                self.torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)
                
            except ImportError:
                raise ImportError(
                    "torch is required for TransformersBackend. "
                    "Install with: pip install torch"
                )
        
        if self._transformers is None:
            try:
                import transformers
                self._transformers = transformers
            except ImportError:
                raise ImportError(
                    "transformers is required for TransformersBackend. "
                    "Install with: pip install transformers"
                )
    
    def _get_hf_model_name(self, model_name: str) -> str:
        """Convert Ollama-style name to HuggingFace name, or return as-is."""
        # Check mapping first
        if model_name in self.model_mappings:
            return self.model_mappings[model_name]
        
        # If it looks like a HuggingFace path, return as-is
        if '/' in model_name:
            return model_name
        
        # Return original if not mapped
        return model_name
    
    def _load_model(self, model_name: str):
        """Load a model and tokenizer if not already loaded."""
        self._ensure_imports()
        
        hf_name = self._get_hf_model_name(model_name)
        
        if model_name in self._loaded_models:
            return self._loaded_models[model_name], self._loaded_tokenizers[model_name]
        
        print(f"Loading model: {hf_name}...")
        
        # Clear previous models to free VRAM
        if self._loaded_models:
            for key in list(self._loaded_models.keys()):
                del self._loaded_models[key]
                del self._loaded_tokenizers[key]
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        
        AutoTokenizer = self._transformers.AutoTokenizer
        AutoModelForCausalLM = self._transformers.AutoModelForCausalLM
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device,
            "trust_remote_code": True,
        }
        
        # 4-bit quantization
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except ImportError:
                print("Warning: bitsandbytes not installed, skipping 4-bit quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)
        model.eval()
        
        self._loaded_models[model_name] = model
        self._loaded_tokenizers[model_name] = tokenizer
        
        # Print VRAM usage
        if self._torch.cuda.is_available():
            allocated = self._torch.cuda.memory_allocated() / 1e9
            print(f"Model loaded. VRAM usage: {allocated:.2f} GB")
        
        return model, tokenizer
    
    def get_available_models(self) -> List[str]:
        """Return list of mapped model names."""
        return list(self.model_mappings.keys())
    
    def generate(self, model_name: str, prompt: str, timeout: int = 300) -> Dict[str, Any]:
        """Generate using Transformers model."""
        import time
        
        start_time = time.time()
        response_text = ""
        success = True
        error_type = None
        
        try:
            model, tokenizer = self._load_model(model_name)
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with self._torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode (only new tokens)
            input_length = inputs['input_ids'].shape[1]
            response_text = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            success = False
            error_type = "TRANSFORMERS_ERROR"
        
        duration = time.time() - start_time
        
        return {
            "response": response_text,
            "success": success,
            "error_type": error_type,
            "duration": round(duration, 3)
        }
    
    def get_backend_info(self) -> Dict[str, str]:
        device_info = "unknown"
        if self._torch and self._torch.cuda.is_available():
            device_info = self._torch.cuda.get_device_name(0)
        
        return {
            "backend": "transformers",
            "device": device_info,
            "dtype": self.torch_dtype_str,
            "use_4bit": str(self.use_4bit),
            "max_new_tokens": str(self.max_new_tokens)
        }
    
    def add_model_mapping(self, ollama_name: str, hf_name: str):
        """Add a custom model name mapping."""
        self.model_mappings[ollama_name] = hf_name
    
    def clear_cache(self):
        """Clear VRAM by unloading all models."""
        if self._loaded_models:
            for key in list(self._loaded_models.keys()):
                del self._loaded_models[key]
                del self._loaded_tokenizers[key]
            if self._torch and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            print("Cache cleared.")


class RemoteOllamaBackend(InferenceBackend):
    """
    Backend for remote Ollama server via REST API.
    Useful for connecting to Ollama running on another machine or via ngrok.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize with Ollama server URL.
        
        Args:
            base_url: Full URL to Ollama server (e.g., "http://localhost:11434")
        """
        self.base_url = base_url.rstrip('/')
    
    def get_available_models(self) -> List[str]:
        """Get models from Ollama API."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            print(f"Error fetching remote Ollama models: {e}")
            return []
    
    def generate(self, model_name: str, prompt: str, timeout: int = 300) -> Dict[str, Any]:
        """Generate using Ollama REST API."""
        import time
        
        start_time = time.time()
        response_text = ""
        success = True
        error_type = None
        
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get('response', '')
            
        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            success = False
            error_type = "REMOTE_API_ERROR"
        
        duration = time.time() - start_time
        
        return {
            "response": response_text,
            "success": success,
            "error_type": error_type,
            "duration": round(duration, 3)
        }
    
    def get_backend_info(self) -> Dict[str, str]:
        return {
            "backend": "remote_ollama",
            "base_url": self.base_url
        }


def get_backend(backend_type: str = "auto", **kwargs) -> InferenceBackend:
    """
    Factory function to get appropriate backend.
    
    Args:
        backend_type: "ollama", "transformers", "remote", or "auto"
        **kwargs: Additional arguments for the backend
        
    Returns:
        Configured InferenceBackend instance
    """
    if backend_type == "auto":
        # Auto-detect: use Transformers in Colab, Ollama otherwise
        if IS_COLAB:
            backend_type = "transformers"
        else:
            backend_type = "ollama"
    
    if backend_type == "ollama":
        return OllamaBackend(
            host=kwargs.get('host', 'localhost'),
            port=kwargs.get('port', 11434)
        )
    elif backend_type == "transformers":
        return TransformersBackend(
            device=kwargs.get('device', 'auto'),
            torch_dtype=kwargs.get('torch_dtype', 'float16'),
            use_4bit=kwargs.get('use_4bit', False),
            max_new_tokens=kwargs.get('max_new_tokens', 512)
        )
    elif backend_type == "remote":
        return RemoteOllamaBackend(
            base_url=kwargs.get('base_url', 'http://localhost:11434')
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Quick test
if __name__ == "__main__":
    print("Testing backend detection...")
    print(f"Running in Colab: {IS_COLAB}")
    
    # Test Ollama backend
    backend = get_backend("ollama")
    print(f"\nOllama Backend: {backend.get_backend_info()}")
    models = backend.get_available_models()
    print(f"Available models: {models[:5]}..." if len(models) > 5 else f"Available models: {models}")
    
    print("\nBackends module loaded successfully!")
