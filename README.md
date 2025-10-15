Rwanda Agricultural Chatbot
A domain-specific conversational AI system for Rwanda agriculture, powered by FLAN-T5 and fine-tuned on synthetic Rwanda-specific and general agriculture Q&A datasets. Deployed via Gradio with multi-turn chat capabilities and out-of-domain (OOD) guarding.
Quick Start
Setup

Environment Setup

bash   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
Requires Python 3.12
Data Preparation
Generate and tokenize synthetic datasets:
bashpython3 -m src.data_prep \
  --output_dir data/processed \
  --generate_rwanda 800 \
  --generate_general 4000 \
  --max_input_length 256 \
  --max_target_length 128 \
  --upsample_limit 10
Training
Train the model using TensorFlow (CPU-friendly baseline):
bashCUDA_VISIBLE_DEVICES="" python3 -m src.train \
  --processed_dir data/processed \
  --model_name google/flan-t5-base \
  --output_dir models/run_balanced \
  --batch_size 2 \
  --epochs 2 \
  --learning_rate 5e-5 \
  --optimizer adamw \
  --warmup_ratio 0.06
Evaluation
Evaluate model performance on BLEU, ROUGE, token-F1, and Perplexity metrics:
bashpython3 -m src.evaluate \
  --processed_dir data/processed \
  --model_dir models/run_balanced \
  --num_examples 200 \
  --report_perplexity
Run the Application
Start the Gradio web interface:
bashpython3 -m src.app \
  --model_dir models/run_balanced \
  --host 0.0.0.0 \
  --port 7860
Project Structure
ComponentFilePurposeProject DefinitionREADME.mdDomain justification and project overviewDataset & Preprocessingsrc/data_prep.pyGenerates Rwanda-specific Q&A (crops, seasons, soils, pests), text normalization, and tokenizationModel Fine-tuningsrc/train.pyTensorFlow fine-tuning with AdamW optimizer, learning rate warmup, and checkpoint resumption; logs training_summary.jsonEvaluation Metricssrc/evaluate.pyReports BLEU, ROUGE, token-F1, and optional Perplexity scoresUser Interfacesrc/app.pyMulti-turn conversational chat, OOD detection, and rule-based safety fallbacks
Experiments & Hyperparameters
Compare model configurations by tracking performance:

Learning Rate: 3e-5 vs 5e-5
Optimizer: Adam vs AdamW (with 0.06 warmup)
Batch Size: 2 vs 8 (with smaller model)

Each training run saves detailed metrics to models/<run>/training_summary.json and evaluation outputs for report comparison.
Model Variants

Default: google/flan-t5-base – balanced performance and speed
Lightweight: google/flan-t5-small – faster inference for CPU-only environments

Demo
A 5–10 minute demo video should showcase:

Environment setup and data generation
Model training (short run)
Evaluation metrics output
UI interactions with in-domain and out-of-domain queries
Seasonal agriculture Q&A examples
Key findings and limitations

Notes

CPU-only support: Full training runs on CPU; use google/flan-t5-small for faster iteration
Privacy: Conversation logs are disabled by default; the UI remains clean and lightweight
No external data required: All training data is synthetically generated

Features

Multi-turn Chat: Contextual conversation support for follow-up questions
Out-of-Domain Detection: Identifies and safely handles queries outside agricultural scope
Rule-based Fallbacks: Safety mechanisms for uncertain predictions
Rwanda-Focused: Trained on domain-specific agricultural knowledge for Rwanda


Get started in minutes: Clone the repository, run the setup commands, and launch the Gradio interface to interact with the chatbot!
