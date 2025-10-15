# Rwanda Agricultural Chatbot

> A domain-specific conversational AI system for Rwanda agriculture, powered by FLAN-T5 and fine-tuned on synthetic Rwanda-specific and general agriculture Q&A datasets. Deployed via Gradio with multi-turn chat capabilities and out-of-domain guarding.

## ✨ Features

- **🌾 Rwanda-Focused**: Fine-tuned on 800 synthetic Rwanda agricultural Q&A pairs covering crops, seasons, soils, and pests
- **💬 Multi-turn Chat**: Contextual conversation support for follow-up questions
- **🛡️ OOD Detection**: Identifies and safely handles queries outside agricultural scope with rule-based fallbacks
- **⚡ CPU-Friendly**: Full training and inference support on CPU with option for GPU acceleration
- **📊 Comprehensive Metrics**: BLEU, ROUGE, token-F1, and Perplexity evaluation
- **🎯 Lightweight**: Clean, minimal UI with conversation logging disabled by default

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- pip

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd rwanda-agri-chatbot
```

2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Data Preparation

Generate synthetic training datasets (no external CSVs required):

```bash
python3 -m src.data_prep \
  --output_dir data/processed \
  --generate_rwanda 800 \
  --generate_general 4000 \
  --max_input_length 256 \
  --max_target_length 128 \
  --upsample_limit 10
```

This creates:
- **800** Rwanda-specific Q&A pairs
- **4000** general agriculture Q&A pairs
- Normalized and tokenized datasets ready for training

## 🏋️ Training

### Baseline (CPU-Friendly)

Train the FLAN-T5 base model with recommended hyperparameters:

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

Training outputs are saved to `models/run_balanced/training_summary.json`

### Lightweight Alternative

For faster iteration on CPU, use the smaller variant:

```bash
CUDA_VISIBLE_DEVICES="" python3 -m src.train \
  --model_name google/flan-t5-small \
  --batch_size 8 \
  --output_dir models/run_small
```

## 📈 Evaluation

Evaluate model performance across multiple metrics:

```bash
python3 -m src.evaluate \
  --processed_dir data/processed \
  --model_dir models/run_balanced \
  --num_examples 200 \
  --report_perplexity
```

**Metrics reported:**
- BLEU Score
- ROUGE Score
- Token-F1
- Perplexity

## 🎮 Run the Application

Start the interactive Gradio chat interface:

```bash
python3 -m src.app \
  --model_dir models/run_balanced \
  --host 0.0.0.0 \
  --port 7860
```

Access the chatbot at `http://localhost:7860`

## 📁 Project Structure

```
rwanda-agri-chatbot/
├── src/
│   ├── data_prep.py          # Dataset generation and preprocessing
│   ├── train.py              # Model fine-tuning (TensorFlow)
│   ├── evaluate.py           # Evaluation metrics (BLEU, ROUGE, F1)
│   └── app.py                # Gradio interface
├── data/
│   └── processed/            # Generated tokenized datasets
├── models/
│   └── run_balanced/         # Fine-tuned model checkpoints
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔬 Experiments & Hyperparameters

Track performance across different configurations:

| Configuration | Learning Rate | Optimizer | Batch Size | Notes |
|--------------|---------------|-----------|-----------|-------|
| Baseline | 5e-5 | AdamW | 2 | Recommended |
| Variant 1 | 3e-5 | AdamW | 2 | Lower LR |
| Variant 2 | 5e-5 | Adam | 2 | No warmup |
| Fast | 5e-5 | AdamW | 8 | Small model |

Save results from `models/<run>/training_summary.json` for comparison in your report.

## 💡 Model Information

- **Base Model**: FLAN-T5 (247M parameters)
- **Fine-tuning**: TensorFlow with AdamW optimizer
- **Learning Rate Warmup**: 0.06 ratio
- **Training Data**: 4800 synthetic Q&A pairs

## 🎬 Demo

Create a 5–10 minute demo video showcasing:

1. Environment setup and data generation
2. Model training workflow
3. Evaluation metrics and performance
4. Interactive UI with in-domain queries (crops, seasons, soil management)
5. Out-of-domain handling
6. Key findings and limitations

## 💻 System Requirements

- **CPU**: Works on standard CPUs (tested on 2-4 cores)
- **GPU**: Optional CUDA support for faster training
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~2GB for model and data

## 📝 Notes

- **Synthetic Data**: No external CSV files needed; all data is generated
- **Privacy**: Conversation logs disabled by default for user privacy
- **Scalability**: Easily extend with additional Rwanda-specific Q&A patterns
- **Customization**: Modify `src/data_prep.py` to generate domain-specific content

## 🛠️ Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 1
- Use `google/flan-t5-small` instead of base

**Slow training?**
- Enable GPU: Remove `CUDA_VISIBLE_DEVICES=""`
- Use smaller model variant

**Port already in use?**
- Change `--port` parameter to an available port

## 📄 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built for Rwanda 🇷🇼 | Made with ❤️ for agricultural innovation**
