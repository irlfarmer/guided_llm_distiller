import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU usage initially since we're having CUDA issues
FORCE_CPU = True

def setup_model(model_path):
    """Setup model and tokenizer with proper error handling"""
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with specific settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 instead of float16
            low_cpu_mem_usage=True,
            device_map='auto' if not FORCE_CPU else 'cpu'
        )
        
        # Move to CPU if forced or CUDA unavailable
        device = 'cpu' if FORCE_CPU or not torch.cuda.is_available() else 'cuda'
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_reply(model, tokenizer, prompt, device):
    """Generate reply with robust error handling and numerical stability fixes"""
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation parameters tuned for numerical stability
        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            temperature=0.9,  # Increased temperature for better stability
            top_p=0.95,      # Increased top_p
            top_k=50,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Reduced repetition penalty
            min_length=10,    # Added min_length
            no_repeat_ngram_size=2,  # Prevent repetition of n-grams
            early_stopping=True,
            bad_words_ids=None,  # Disable bad words filtering
            renormalize_logits=True  # Add logit renormalization
        )
        
        # Decode output with extra error checking
        try:
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not full_output.strip():  # Check for empty output
                return "I apologize, but I couldn't generate a meaningful response. Please try again."
            
            # Extract answer if it exists
            if "Answer:" in full_output:
                return full_output.split("Answer:")[-1].strip()
            return full_output.strip()
        except Exception as decode_error:
            print(f"Decoding error: {str(decode_error)}")
            return "I apologize, but I encountered an error processing the response. Please try again."
    
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "I apologize, but I encountered an error generating a response. Please try again."

def main():
    model_path = "./distilled_model"
    print(f"Loading model and tokenizer from {model_path} ...")
    
    try:
        model, tokenizer, device = setup_model(model_path)
        print(f"Model loaded successfully. Running on: {device}")
        
        print("Chatbot is ready! (Type 'quit' or 'exit' to stop)")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting. Bye!")
                break
            if user_input == "":
                continue

            # Format prompt
            prompt = f"Question: {user_input}\nAnswer:"
            
            # Generate response
            reply = generate_reply(model, tokenizer, prompt, device)
            print("Bot:", reply)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return

if __name__ == "__main__":
    main()
