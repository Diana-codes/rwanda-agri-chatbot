# 🌾 Rwanda Agricultural Chatbot Report

**Author:** Diana Ruzindana  
**GitHub Repo:** https://github.com/Diana-codes/rwanda-agri-chatbot  
**Live Demo (HF Space):** https://huggingface.co/spaces/Ruzindana/rwanda-agri-chatbot  
**Model Card (HF Hub):** https://huggingface.co/Ruzindana/rwanda-agri-bot-model  
**Demo Video (5–10 min):** https://youtu.be/W_En_Ny74UE  

## 1. Project Definition & Domain Alignment

**Purpose:** A Rwanda-specific agriculture assistant that answers farmer questions on:
- Planting seasons (Season A/B)
- Soil types and fertility management
- Pest and disease control
- Irrigation practices
- Market access and cooperatives

**Domain Focus:** Localized guidance for Rwanda's key crops: maize, beans, potatoes, cassava, banana, coffee, and tea.

**Relevance:** Smallholder farmers in Rwanda need quick, accurate, and contextually appropriate advice. This chatbot provides rapid, Rwanda-specific responses while safely rejecting out-of-domain queries that could mislead farmers.

## 2. Dataset Collection & Preprocessing

**Data Sources:**
- **Rwanda Agricultural Q&A Dataset** (~800 examples): Crops, planting windows by Season A/B, soil types (sandy/clay/acidic/loamy), pests, irrigation, cooperatives
- **General Agriculture Q&A Dataset** (~4,000 examples): Generic best practices to improve coverage and generalization

**Data Balancing:** Rwanda dataset upsampled (≤10×) to reduce bias toward general agriculture data.

**Preprocessing Pipeline:**
1. **Text Normalization:** Lowercasing, whitespace cleanup, contraction expansion
2. **Prompt Engineering:** Instruction prefix biasing answers toward Rwanda conditions and Season A/B timing
3. **Tokenization:** FLAN‑T5 tokenizer with max input length 256, target length 128
4. **Data Split:** 80% train, 10% validation, 10% test

**Reproducible Command:**
```bash
python3 -m src.data_prep \
    --output_dir data/processed \
    --generate_rwanda 800 \
    --generate_general 4000 \
    --max_input_length 256 \
    --max_target_length 128 \
    --upsample_limit 10
```

## 3. Model & Fine-Tuning

**Architecture:** FLAN‑T5 encoder-decoder Transformer implemented in TensorFlow  
**Base Model:** google/flan-t5-base (fallback: google/flan-t5-small for resource constraints)  
**Objective:** Sequence-to-sequence conditional generation on instruction-style prompts

**Hyperparameter Experiments:**

| Run | Optimizer | Learning Rate | Warmup | Batch Size | Epochs | Val Loss |
|-----|-----------|---------------|--------|------------|--------|----------|
| base_adam | Adam | 5e-5 | 0.00 | 2 | 2 | 0.0087 |
| adamw_warm | AdamW | 5e-5 | 0.06 | 2 | 2 | 0.0072 |

**Training Command:**
```bash
CUDA_VISIBLE_DEVICES="" python3 -m src.train \
    --processed_dir data/processed \
    --model_name google/flan-t5-base \
    --output_dir models/run_balanced \
    --batch_size 2 \
    --epochs 2 \
    --learning_rate 5e-5 \
    --optimizer adamw \
    --warmup_ratio 0.06
```

## 4. Evaluation & Results

**Metrics Used:** BLEU, ROUGE‑1/2/L, Token-level Precision/Recall/F1, Perplexity

**Quantitative Results:**
- **BLEU Score:** 0.42
- **ROUGE‑L:** 0.38 
- **Token-F1:** 0.68
- **Perplexity:** 2.8

**Evaluation Command:**
```bash
python3 -m src.evaluate \
    --processed_dir data/processed \
    --model_dir models/run_balanced \
    --num_examples 200 \
    --report_perplexity
```

**Qualitative Observations:**
- Successfully answers Rwanda seasonal questions with Season A/B timing
- Provides soil and fertility management advice with IPM pest control tips
- Politely rejects out-of-domain queries using domain filter and rule-based fallbacks

**Key Findings:**
- Rwanda-grounded prompts + upsampling significantly reduced generic/global answers
- AdamW + warmup stabilized training and improved validation loss (0.0087 → 0.0072)
- Rule-based fallbacks for critical intents (planting seasons, soil suitability) improved answer consistency

**Limitations:**
- Synthetic Q&A may miss micro-regional variations within Rwanda
- CPU-only training limits model size; smaller models trade some fluency for speed

## 5. UI & Deployment

**Interface:** Gradio multi-turn chat with clean, intuitive design and right-sidebar layout

**Safety & Robustness Features:**
- Out-of-domain guard with agriculture keyword detection
- Rule-based fallbacks for critical intents (planting seasons, soil suitability)
- Few-shot examples in prompts for better steering

**Deployment:**
- **Live Demo:** https://huggingface.co/spaces/Ruzindana/rwanda-agri-chatbot
- **Local Launch:** `python3 -m src.app --model_dir models/run_balanced --host 0.0.0.0 --port 7860`

## 6. Reproducibility

**Environment Requirements:**
- Python 3.12
- TensorFlow 2.17–2.20
- Transformers 4.40–4.43
- tf-keras, numpy <2, gradio

**Reproduction Steps:**
1. Setup virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Generate data (Section 2)
4. Train model (Section 3)
5. Evaluate performance (Section 4)
6. Run app locally or deploy to HF Spaces

**Model Availability:** Hosted at Hugging Face Model Hub, loaded by model ID in the Space

## 7. Ethical Considerations

- **Data Privacy:** Synthetic, non-PII data used throughout
- **Safety:** Encourages verification with local RAB extension services and soil testing
- **Scope Limitation:** OOD guard prevents unsafe or irrelevant advice outside agriculture domain

## 8. Contributions (Individual)

- **Data Engineering:** Designed and implemented synthetic Q&A generation patterns with Rwanda-specific content
- **Model Training:** Conducted hyperparameter experiments and implemented resume functionality
- **Evaluation:** Built comprehensive evaluation pipeline with multiple NLP metrics
- **UI Development:** Created intuitive Gradio interface with conversation history and safety features
- **Deployment:** Successfully deployed to Hugging Face Spaces with model hosting on HF Hub
- **Documentation:** Prepared comprehensive README, report, and code documentation

## 9. References

- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)"
- Hugging Face Transformers, Datasets, and Gradio documentation
- TensorFlow Keras documentation

## 10. Appendix

**Example Conversations:**

| Question | Answer |
|----------|--------|
| When should I plant maize in Rwanda? | Plant at the start of Season B (Mar–Jun) when rains begin. Use varieties like ZM607. |
| What soil is good for beans? | Well-drained loamy soils; add compost and avoid waterlogging. |
| Which crops grow well in sandy soils? | Maize, cassava, and groundnuts with added organic matter and mulch. |
| What is the capital of Rwanda? | I'm focused on agriculture. Please ask about crops, soils, pests, seasons, or farming practices. |

**Training Logs:** Available in `models/run_balanced/training_summary.json`
```
