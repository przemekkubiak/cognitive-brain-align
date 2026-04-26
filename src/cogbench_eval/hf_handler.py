"""
Custom HuggingFace LLM handler for CogBench.

This module provides a HuggingFace model handler that properly supports
EleutherAI/Pythia models and other open-source LLMs.

To use this with CogBench:
1. Copy this file to CogBench/llm_utils/eleutherai.py
2. Register it in CogBench/llm_utils/llms.py
"""

import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class HFModelHandler:
    """Handler for HuggingFace models compatible with CogBench."""
    
    def __init__(self, model_id, max_tokens=100, temperature=0.7):
        """
        Initialize HuggingFace model handler.
        
        Args:
            model_id: HuggingFace model ID (e.g., 'EleutherAI/pythia-70m')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"Loading model: {model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            
            # Set pad token if not already set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.pipe = transformers.pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_tokens,
            )
            
            print(f"[OK] Successfully loaded {model_id}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt, **kwargs):
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Generated text
        """
        try:
            output = self.pipe(
                prompt,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                **kwargs
            )
            
            # Extract generated text
            generated_text = output[0]['generated_text']
            # Remove the prompt from the output
            return generated_text[len(prompt):]
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise


def get_hf_model(model_id, max_tokens=100, temperature=0.7):
    """
    Factory function to get a HuggingFace model handler.
    
    Args:
        model_id: HuggingFace model ID
        max_tokens: Maximum tokens
        temperature: Temperature
        
    Returns:
        HFModelHandler instance
    """
    return HFModelHandler(model_id, max_tokens, temperature)
