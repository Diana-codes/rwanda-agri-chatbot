import argparse
import gradio as gr
import numpy as np
from typing import Optional, List, Tuple
import json
import os
from datetime import datetime

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max_generate_tokens", type=int, default=96)
    return parser.parse_args()


AGRI_KEYWORDS = [
    "crop", "soil", "fertilizer", "pest", "harvest", "plant", "season",
    "maize", "beans", "coffee", "tea", "potatoes", "cassava", "irrigation",
    "terrace", "rain", "variety", "seed", "compost", "mulch", "market",
    "livestock", "cow", "goat", "chicken", "pig", "storage", "land",
    "climate", "fertility", "price", "drought", "water", "dry season"
]

CROP_KEYWORDS = ["maize", "beans", "coffee", "tea", "potatoes", "cassava", "banana"]
SOIL_KEYWORDS = ["sandy", "clay", "acidic", "loamy", "pH", "drainage", "terrace"]

EXEMPLARS = [
    #  Crop planting seasons
    (
        "when should i plant maize in rwanda?",
        "plant maize at onset of rains: Season A (Sept–Dec) or Season B (Mar–Jun). choose based on local rainfall and soil moisture."
    ),
    (
        "when should i plant beans in rwanda?",
        "plant beans at the start of Season A or Season B rains. for climbing beans use stakes and ensure drainage on slopes."
    ),
    (
        "which crop is best for sandy soils in rwanda?",
        "maize, cassava, and groundnuts can do well if you add organic matter and mulch to retain moisture."
    ),

    #  Land preparation & soil fertility
    (
        "how do farmers in rwanda prepare their land for planting?",
        "farmers usually clear the land, plow or dig, make ridges or terraces on slopes, and add compost or manure before planting."
    ),
    (
        "what can farmers do to improve soil fertility?",
        "they can add compost or manure, practice crop rotation, use cover crops, and avoid over-cultivating the same land."
    ),

    #  Livestock
    (
        "what are the common livestock in rwanda?",
        "common livestock include cows, goats, pigs, chickens, and rabbits. dairy cows are especially important in rural households."
    ),
    (
        "how can farmers take care of dairy cows?",
        "they provide clean water, good feed, shelter, regular vaccination, and hygiene to improve milk production."
    ),

    #  Irrigation & climate
    (
        "how do farmers water crops during dry season?",
        "farmers use small-scale irrigation like watering cans, sprinklers, or gravity-fed systems to keep crops alive during dry spells."
    ),
    (
        "what can farmers do during drought?",
        "they can use mulch to keep moisture, plant drought-tolerant crops, and use water-saving irrigation methods."
    ),

    #  Pest control
    (
        "how can farmers control pests in maize?",
        "they monitor early, use resistant varieties, apply recommended pesticides carefully, and remove affected plants."
    ),

    #  Harvesting & storage
    (
        "how can farmers store their harvest in rwanda?",
        "they dry crops properly, store in clean bags or silos off the ground, and keep storage areas cool and dry to avoid pests."
    ),

    #  Market access
    (
        "how can farmers get better prices for their crops?",
        "they can join cooperatives, sell in bulk, use digital platforms, and time their sales when prices are higher."
    ),
    (
        "where can farmers sell their produce in rwanda?",
        "farmers sell through local markets, cooperatives, collection centers, and agribusiness platforms."
    ),

    #  Climate change
    (
        "how is climate change affecting farming in rwanda?",
        "climate change brings unpredictable rainfall, droughts, and pests. farmers adapt by using improved seeds, irrigation, and mulching."
    )
]


def is_in_domain(question: str) -> bool:
    text = (question or "").lower()
    return any(k in text for k in AGRI_KEYWORDS)


def fallback_answer_for(question: str) -> str:
    q = (question or "").lower()
    if "maize" in q and ("when" in q or "season" in q or "plant" in q):
        return (
            "plant maize at onset of rains: Season A (Sept–Dec) or Season B (Mar–Jun). "
            "pick the season with reliable local rainfall and prepare soil with compost."
        )
    if "beans" in q and ("when" in q or "season" in q or "plant" in q):
        return (
            "plant beans at the start of Season A or B rains. use climbing stakes on slopes and ensure good drainage."
        )
    return "please ask about crops, soil, pests, seasons, or farming practices in rwanda."


def rule_based_reply(message: str) -> str:
    q = (message or "").lower()

    #  Crop planting season
    if ("when" in q or "best season" in q or "season" in q) and any(c in q for c in CROP_KEYWORDS):
        if "maize" in q:
            return "Season A (Sept–Dec) or Season B (Mar–Jun) at onset of rains. Prepare soil with compost; monitor fall armyworm early."
        if "beans" in q:
            return "Start at the first rains in Season A or B. Use climbing stakes on slopes and ensure drainage."
        if "potato" in q or "potatoes" in q:
            return "Plant certified seed before main rains; hill twice and ensure terrace drainage."
        if "cassava" in q:
            return "At onset of rains; use clean cuttings and 1×1 m spacing; manage mosaic risk."
        if "coffee" in q:
            return "At start of short rains (Season A: Sept–Dec); mulch and provide shade; control pests."
        if "tea" in q:
            return "During reliable rains; maintain spacing and frequent nitrogen in small doses."
        return "Plant at onset of Season A or B rains depending on local rainfall; prepare soil and use recommended varieties."

    #  Soil - best crop matching
    if ("which crop" in q and "best" in q) and any(s in q for s in SOIL_KEYWORDS):
        if "sandy" in q:
            return "maize, cassava and groundnuts do well if you add organic matter and mulch."
        if "clay" in q:
            return "beans, rice and vegetables work if you improve drainage and add compost."
        if "acidic" in q:
            return "lime first after soil test; then maize or beans perform better."
        if "loam" in q or "loamy" in q:
            return "most crops including maize and beans thrive; maintain with compost."

    #  Land prep
    if "prepare" in q and "land" in q:
        return "farmers usually clear the land, plow or dig, make ridges or terraces, and add compost or manure before planting."

    #  Storage
    if "store" in q and "harvest" in q:
        return "dry crops well, store in clean bags or silos off the ground, and keep storage areas cool and dry."

    #  Livestock
    if "livestock" in q or "animals" in q:
        return "common livestock in Rwanda are cows, goats, pigs, chickens, and rabbits."

    #  Irrigation
    if "irrigation" in q or ("water" in q and "dry" in q):
        return "farmers use watering cans, sprinklers, or small irrigation systems to keep crops alive in dry periods."

    #  Pest
    if "pest" in q and "maize" in q:
        return "monitor early, use resistant varieties, apply recommended pesticides carefully, and remove affected plants."

    #  Fertility
    if "soil" in q and "fertility" in q:
        return "add compost or manure, rotate crops, use cover crops, and avoid over-cultivation."

    #  Market
    if "market" in q or "price" in q or "sell" in q:
        return "farmers can join cooperatives, sell in bulk, or use digital platforms to get better prices."

    #  Climate
    if "climate" in q or "drought" in q:
        return "farmers adapt to climate change by using improved seeds, irrigation, and mulching."

    return ""


def build_pipeline(model_dir: str, max_generate_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)

    def predict(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        if not message or not message.strip():
            return history, ""
        if not is_in_domain(message):
            reply = (
                "Hello, I am an agriculture assistant. "
                "Please ask about crops, soil, pests, seasons, or farming practices to help you much better."
            )
            history = history + [(message, reply)]
            return history, ""

        # First, try a rule-based reply for critical intents to avoid generic outputs
        rb = rule_based_reply(message)
        if rb:
            history = history + [(message, rb)]
            return history, ""

        # Build brief conversational context from last 3 exchanges
        recent = history[-3:]
        turns = []
        for user_q, bot_a in recent:
            turns.append(f"User: {user_q}")
            turns.append(f"Assistant: {bot_a}")
        context = " \n".join(turns)

        system = (
            "You are an agriculture assistant for farmers in Rwanda. "
            "Give a short, practical answer for Rwanda conditions. "
            "If timing is asked, consider Rwanda Season A (Sep-Dec rains) and Season B (Mar-Jun rains). "
            "If unsure, advise to check local rainfall or extension officer."
        )
        # Few-shot examples block
        ex_lines = []
        for q_ex, a_ex in EXEMPLARS:
            ex_lines.append(f"Q: {q_ex}\nA: {a_ex}")
        examples_block = "\n\n".join(ex_lines)

        prompt = (
            f"{system}\n"
            f"Examples:\n{examples_block}\n\n"
            f"Context:\n{context}\n"
            f"Question: {message.strip()}\n"
            f"Answer:"
        )
        inputs = tokenizer([prompt], return_tensors="np")
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_generate_tokens,
            do_sample=True,
            num_beams=1,
            top_k=50,
            top_p=0.92,
            temperature=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            length_penalty=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        reply = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        generic_bad = ["harvest at physiological maturity"]
        if any(phrase in reply.lower() for phrase in generic_bad) or len(reply.strip()) < 6:
            reply = fallback_answer_for(message)
        # Log conversation for analytics
        log_conversation(message, reply)
        history = history + [(message, reply)]
        return history, ""

    return predict


def log_conversation(question, answer):
    try:
        log_file = "conversation_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = {"questions": [], "answers": [], "timestamps": []}
        
        data["questions"].append(question)
        data["answers"].append(answer)
        data["timestamps"].append(datetime.now().isoformat())
        
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

def get_conversation_stats():
    try:
        if os.path.exists("conversation_log.json"):
            with open("conversation_log.json", "r") as f:
                data = json.load(f)
            total_qs = len(data.get("questions", []))
            topics = {}
            for q in data.get("questions", []):
                for keyword in ["maize", "beans", "coffee", "soil", "pest", "irrigation", "livestock", "market"]:
                    if keyword in q.lower():
                        topics[keyword] = topics.get(keyword, 0) + 1
            return f"📊 **Session Stats:**\n• Total Questions: {total_qs}\n• Popular Topics: {', '.join([f'{k}({v})' for k, v in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]])}"
    except:
        pass
    return "📊 **Session Stats:**\n• Start asking questions to see analytics!"

def main():
    args = parse_args()
    predict = build_pipeline(args.model_dir, args.max_generate_tokens)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                gr.Markdown("# 🌾 Rwanda Agricultural Chatbot")
                gr.Markdown(
                    "Ask any question about farming in Rwanda — crops, soil, pests, irrigation, livestock, or market access."
                )
                chatbot = gr.Chatbot(height=400, show_label=False)
                msg = gr.Textbox(label="Your message", placeholder="When should I plant maize in Season A?", lines=2)
                with gr.Row():
                    clear = gr.Button("Clear Chat", variant="secondary")
                    send = gr.Button("Send", variant="primary")
                state = gr.State(value=[])

            # Empty sidebar on the right
            with gr.Column(scale=1):
                gr.Markdown("")

        def ui_respond(user_message, chat_history):
            updated_history, _ = predict(user_message, chat_history or [])
            return updated_history, "", updated_history

        def clear_chat():
            return [], "", []

        msg.submit(ui_respond, [msg, state], [chatbot, msg, state])
        send.click(ui_respond, [msg, state], [chatbot, msg, state])
        clear.click(clear_chat, None, [chatbot, msg, state])

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
