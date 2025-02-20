import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class LlamaWrapper:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        
        # Special handling for distilled model
        if model_name == "./final_distilled_model":
            # Load the base tokenizer from the original student model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat",
                use_fast=False
            )
            # Then load the distilled model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Regular loading for other models
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # Ensure proper tokenizer configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set proper decoding parameters
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        self.model.eval()
    
    @torch.no_grad()
    def generate_response(self, question: str, max_length: int = 512) -> str:
        """Generate response with more consistent formatting"""
        # Format input with clear separator
        input_text = f"Question: {question.strip()}\nAnswer: "
        
        # Tokenize with explicit parameters
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).to(self.device)
        
        try:
            # Generate with more controlled parameters
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                min_length=32,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                use_cache=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.1,
                early_stopping=True
            )
            
            # Decode with careful cleanup
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up the response more thoroughly
            response = response.replace(input_text, "")  # Remove the input text
            response = response.replace("Question:", "").replace("Answer:", "")  # Remove any repeated markers
            response = response.replace(":", "").strip()  # Remove stray colons
            response = ' '.join(response.split())  # Normalize whitespace
            
            # Handle empty responses
            if not response.strip():
                return "I apologize, but I couldn't generate a meaningful response."
            
            return response
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return "An error occurred while generating the response."

@st.cache_resource
def load_single_model(model_choice: str) -> Optional[LlamaWrapper]:
    """Load a single model with caching"""
    try:
        if model_choice == "Teacher (Llama-2-7B)":
            return LlamaWrapper("meta-llama/Llama-2-7b")
        elif model_choice == "Student (Llama-2-7B-chat)":
            return LlamaWrapper("meta-llama/Llama-2-7b-chat")
        elif model_choice == "Distilled Model":
            return LlamaWrapper("./final_distilled_model")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Llama Model Comparison Interface")
    st.write("Compare responses from different Llama-2 models")
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.write(f"Using device: {device} with {gpu_memory:.2f}GB memory")
    else:
        st.sidebar.write(f"Using device: {device}")
    
    # Memory management tips
    st.sidebar.write("---")
    st.sidebar.write("💡 Memory Management Tips:")
    st.sidebar.write("1. Load one model at a time")
    st.sidebar.write("2. Use shorter inputs for better performance")
    st.sidebar.write("3. Adjust max length if needed")
    
    # Model selector
    model_choice = st.radio(
        "Select model to test:",
        ["Teacher (Llama-2-7B)", "Student (Llama-2-7B-chat)", "Distilled Model"],
        help="Choose one model to load and test"
    )
    
    # Load the selected model
    model = load_single_model(model_choice)
    
    if model is None:
        st.error("Failed to load model. Please check the errors above.")
        return
    
    # Input area
    question = st.text_area(
        "Enter your question:",
        height=100,
        help="Enter your question here"
    )
    
    # Generation parameters
    with st.expander("Advanced Settings"):
        max_length = st.slider(
            "Max response length",
            min_value=64,
            max_value=512,
            value=256,
            help="Adjust the maximum length of the response"
        )
    
    if st.button("Generate Response"):
        if not question:
            st.warning("Please enter a question.")
            return
        
        try:
            with st.spinner(f"Generating response using {model_choice}..."):
                response = model.generate_response(question, max_length=max_length)
            
            # Display response in a nice format
            st.subheader("Response:")
            st.write(response)
            
            # Add some metrics
            st.write("---")
            st.write(f"Response length: {len(response.split())} words")
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main() 