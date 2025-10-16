# Rwanda Agricultural Chatbot - Hugging Face Spaces

This is the deployment version for Hugging Face Spaces.

## Quick Start

1. **Fork this repository** to your Hugging Face account
2. **Create a new Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose "Gradio" as SDK
   - Name it `rwanda-agri-bot` (or your preferred name)
3. **Upload files**:
   - Upload `app.py` as the main file
   - Upload `requirements.txt`
   - Upload your trained model from `models/run_balanced/` to the Space
4. **Deploy**: The Space will automatically build and deploy

## Files Structure for HF Spaces

```
your-space/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── models/
│   └── run_balanced/     # Your trained model files
│       ├── config.json
│       ├── tf_model.h5
│       ├── tokenizer.json
│       └── ...
└── README.md             # This file
```

## Model Upload

To upload your trained model to HF Spaces:

1. **Compress your model**:
   ```bash
   cd models
   tar -czf run_balanced.tar.gz run_balanced/
   ```

2. **Upload to Space**:
   - Go to your Space's "Files" tab
   - Upload `run_balanced.tar.gz`
   - Extract it in the Space

3. **Update app.py** if needed:
   - Change `MODEL_DIR = "models/run_balanced"` to match your model path

## Features

- 🌾 Rwanda-specific agricultural advice
- 💬 Multi-turn conversation history
- 🎯 Domain-specific question detection
- 📱 Mobile-friendly interface
- 🔄 Real-time responses

## Demo Questions

Try these sample questions:
- "When should I plant maize in Rwanda Season A?"
- "What soil is good for growing beans?"
- "How do I control fall armyworm in maize?"
- "What crops grow well in sandy soils?"
- "How do cooperatives help farmers in Rwanda?"

## Troubleshooting

- **Model not loading**: Check that model files are in the correct directory
- **Memory issues**: The app uses CPU-only inference for HF Spaces compatibility
- **Slow responses**: This is normal for CPU inference on free tier

## Local Testing

To test locally before deploying:

```bash
python app.py
```

Then visit `http://localhost:7860`
