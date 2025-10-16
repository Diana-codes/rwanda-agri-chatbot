#!/usr/bin/env python3
"""
Rwanda Agricultural Chatbot - Hugging Face Spaces Deployment
"""

import gradio as gr
import os
import json
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Configuration for Hugging Face Spaces
MODEL_DIR = "models/run_balanced"  # Will be uploaded to HF Space
MAX_GENERATE_TOKENS = 150

# Agriculture keywords for domain detection
AGRI_KEYWORDS = {
    "crops": ["maize", "beans", "potato", "cassava", "banana", "coffee", "tea", "rice", "wheat", "sorghum"],
    "seasons": ["season a", "season b", "planting", "harvest", "rainy", "dry"],
    "soil": ["soil", "fertile", "drainage", "ph", "compost", "manure", "lime"],
    "pests": ["pest", "disease", "blight", "mosaic", "armyworm", "borer"],
    "practices": ["irrigation", "fertilizer", "cooperative", "market", "storage", "terrace"]
}

# Few-shot examples for better steering
EXEMPLARS = [
    ("When should I plant maize in Rwanda?", "Plant maize at the start of Season B (Mar-Jun) when rains begin. Use improved varieties like ZM607."),
    ("What soil is good for beans?", "Beans grow well in loamy soils with good drainage. Add compost and avoid waterlogging."),
    ("How to control fall armyworm?", "Handpick larvae early, use neem extract, or apply Bt spray. Monitor regularly during wet season.")
]

def is_in_domain(text: str) -> bool:
    """Check if query is agriculture-related"""
    text_lower = text.lower()
    return any(keyword in text_lower for keywords in AGRI_KEYWORDS.values() for keyword in keywords)

def fallback_answer_for(question: str) -> str:
    """Provide fallback answers for common questions"""
    q_lower = question.lower()
    
    if "season" in q_lower and "plant" in q_lower:
        if "maize" in q_lower:
            return "Plant maize at the start of Season B (Mar-Jun) when rains begin. Use improved varieties like ZM607 or SC513."
        elif "bean" in q_lower:
            return "Plant beans at the onset of rains in Season A (Sept-Dec) or Season B (Mar-Jun). Use climbing varieties on stakes."
        elif "potato" in q_lower:
            return "Plant potatoes during Season A (Sept-Dec) or Season B (Mar-Jun). Use certified seed potatoes like Kinigi or Victoria."
        else:
            return "In Rwanda, plant most crops at the start of Season A (Sept-Dec) or Season B (Mar-Jun) when rains begin."
    
    elif "soil" in q_lower and ("good" in q_lower or "suitable" in q_lower):
        if "maize" in q_lower:
            return "Maize grows well in loamy soils with good drainage. Add compost and test soil pH (should be 5.5-7.0)."
        elif "bean" in q_lower:
            return "Beans prefer well-drained loamy or clay soils. Improve drainage with raised beds if needed."
        else:
            return "Most crops grow best in loamy soils with good drainage. Test your soil pH and add organic matter."
    
    elif "pest" in q_lower or "disease" in q_lower:
        return "For pest control, use integrated pest management: crop rotation, handpicking, neem extract, or approved pesticides. Contact RAB for specific advice."
    
    elif "fertilizer" in q_lower or "manure" in q_lower:
        return "Apply organic compost and balanced NPK fertilizer based on soil test. RAB provides soil testing services."
    
    elif "cooperative" in q_lower:
        return "Cooperatives help with market access, bulk buying, training, and fair prices. Contact your local cooperative or RAB."
    
    return None

def build_pipeline(model_dir: str, max_tokens: int):
    """Build the prediction pipeline"""
    try:
        # Load model and tokenizer
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        def predict(question: str, chat_history: list) -> tuple:
            """Generate response for a question"""
            if not question.strip():
                return chat_history, chat_history
            
            # Check if in domain
            if not is_in_domain(question):
                response = "I'm an agriculture assistant for Rwanda. Please ask about crops, soil, pests, seasons, or farming practices."
                new_history = chat_history + [(question, response)]
                return new_history, new_history
            
            # Try rule-based fallback first
            fallback = fallback_answer_for(question)
            if fallback:
                new_history = chat_history + [(question, fallback)]
                return new_history, new_history
            
            # Build prompt with few-shot examples
            prompt = "You are an agriculture assistant for smallholder farmers in Rwanda. Answer briefly and specifically for Rwanda conditions.\n\n"
            
            # Add few-shot examples
            for ex_q, ex_a in EXEMPLARS:
                prompt += f"Q: {ex_q}\nA: {ex_a}\n\n"
            
            prompt += f"Q: {question}\nA:"
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="tf", max_length=512, truncation=True)
            
            with tf.device('/CPU:0'):  # Force CPU for HF Spaces
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_tokens,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    length_penalty=1.2
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if "A:" in response:
                response = response.split("A:")[-1].strip()
            
            if not response or len(response) < 10:
                response = "I need more specific information about your farming question. Please provide more details."
            
            new_history = chat_history + [(question, response)]
            return new_history, new_history
        
        return predict
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback function
        def fallback_predict(question: str, chat_history: list) -> tuple:
            response = "Model loading error. Please try again later."
            new_history = chat_history + [(question, response)]
            return new_history, new_history
        return fallback_predict

def main():
    """Main Gradio interface"""
    # Initialize prediction function
    predict = build_pipeline(MODEL_DIR, MAX_GENERATE_TOKENS)
    
    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Rwanda Agricultural Chatbot") as demo:
        gr.Markdown("# 🌾 Rwanda Agricultural Chatbot")
        gr.Markdown(
            "Ask any question about farming in Rwanda — crops, soil, pests, irrigation, livestock, or market access."
        )
        
        chatbot = gr.Chatbot(height=400, show_label=False)
        msg = gr.Textbox(
            label="Your message", 
            placeholder="When should I plant maize in Season A?", 
            lines=2
        )
        
        with gr.Row():
            clear = gr.Button("Clear Chat", variant="secondary")
            send = gr.Button("Send", variant="primary")
        
        state = gr.State(value=[])
        
        def ui_respond(user_message, chat_history):
            updated_history, _ = predict(user_message, chat_history or [])
            return updated_history, "", updated_history
        
        def clear_chat():
            return [], "", []
        
        msg.submit(ui_respond, [msg, state], [chatbot, msg, state])
        send.click(ui_respond, [msg, state], [chatbot, msg, state])
        clear.click(clear_chat, None, [chatbot, msg, state])
    
    # Launch with public sharing for HF Spaces
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default HF Spaces port
        share=False,            # HF Spaces handles sharing
        show_error=True
    )

if __name__ == "__main__":
    main()
