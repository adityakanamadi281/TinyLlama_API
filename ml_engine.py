import torch
from transformers import pipeline

class LLMEngine:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        """Loads the model into memory. This happens only once."""
        print("⏳ Loading TinyLlama... (This might take a minute)")
        
        # We use 'pipeline' for simplicity. 
        # model_id can be swapped for other small models like 'distilgpt2'
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16, # Saves memory
            device_map="auto" # Uses GPU if available, otherwise CPU
        )
        print("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: int = 256):
        """Generates text based on the prompt."""
        if not self.pipe:
            raise RuntimeError("Model is not loaded!")
            
        # Create a chat-like format for TinyLlama
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        
        # Apply the chat template
        prompt_formatted = self.pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt_formatted, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95
        )
        
        # Clean up the output to return just the generated text
        generated_text = outputs[0]["generated_text"]
        # Remove the prompt from the response to be cleaner
        return generated_text[len(prompt_formatted):]

# Create a global instance
llm_engine = LLMEngine()