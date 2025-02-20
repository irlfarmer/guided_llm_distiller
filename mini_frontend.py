import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU usage initially since we're having CUDA issues
FORCE_CPU = True

def setup_model(model_path):
    """Setup model and tokenizer with proper error handling."""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for numerical stability
            low_cpu_mem_usage=True,
            device_map='auto' if not FORCE_CPU else 'cpu'
        )
        
        # Determine device
        device = 'cpu' if FORCE_CPU or not torch.cuda.is_available() else 'cuda'
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_reply(model, tokenizer, prompt, device):
    """Generate a reply using model.generate with stability fixes."""
    try:
        # Prepare input tokens
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Use no_grad to avoid gradient tracking during generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                top_k=50,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                min_length=10,
                no_repeat_ngram_size=2,
                early_stopping=False  # Set to False for sampling mode
                # renormalize_logits removed to avoid numerical issues
            )
        
        # Decode the output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not full_output.strip():
            return "I apologize, but I couldn't generate a meaningful response. Please try again."
        
        # Optionally, if your prompt expects an "Answer:" section, extract it.
        if "Answer:" in full_output:
            return full_output.split("Answer:")[-1].strip()
        return full_output.strip()
    
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "I apologize, but I encountered an error generating a response. Please try again."

def main():
    model_path = "./final_distilled_model"
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
