#!/usr/bin/env python3
"""
Script to prepare files for Hugging Face Spaces deployment
"""

import os
import shutil
import tarfile
from pathlib import Path

def prepare_for_hf_spaces():
    """Prepare files for Hugging Face Spaces deployment"""
    
    # Create deployment directory
    deploy_dir = Path("hf_deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    print("🚀 Preparing files for Hugging Face Spaces deployment...")
    
    # Copy main files
    files_to_copy = [
        "app.py",
        "requirements.txt", 
        "README_HF_Spaces.md"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir / file)
            print(f"✅ Copied {file}")
        else:
            print(f"❌ Missing {file}")
    
    # Create models directory and copy model
    models_dir = deploy_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_source = "models/run_balanced"
    if os.path.exists(model_source):
        # Copy model files
        shutil.copytree(model_source, models_dir / "run_balanced", dirs_exist_ok=True)
        print(f"✅ Copied model from {model_source}")
        
        # Create compressed version for easy upload
        with tarfile.open(deploy_dir / "run_balanced.tar.gz", "w:gz") as tar:
            tar.add(model_source, arcname="run_balanced")
        print("✅ Created run_balanced.tar.gz for easy upload")
    else:
        print(f"❌ Model not found at {model_source}")
        print("   Make sure you've trained a model first!")
    
    # Create deployment instructions
    instructions = """
# Hugging Face Spaces Deployment Instructions

## Step 1: Create Space
1. Go to https://huggingface.co/new-space
2. Choose "Gradio" as SDK
3. Name it "rwanda-agri-bot" (or your preferred name)
4. Set to Public

## Step 2: Upload Files
1. Upload these files to your Space:
   - app.py
   - requirements.txt
   - README_HF_Spaces.md

## Step 3: Upload Model
1. Upload run_balanced.tar.gz to your Space
2. Extract it in the Space (or upload the model files directly)

## Step 4: Deploy
1. The Space will automatically build and deploy
2. Wait for the build to complete (usually 2-5 minutes)
3. Your chatbot will be live at: https://yourusername-rwanda-agri-bot.hf.space

## Step 5: Test
1. Try sample questions:
   - "When should I plant maize in Rwanda Season A?"
   - "What soil is good for growing beans?"
   - "How do I control fall armyworm?"

## Troubleshooting
- If model doesn't load: Check file paths in app.py
- If build fails: Check requirements.txt versions
- If slow: Normal for CPU inference on free tier
"""
    
    with open(deploy_dir / "DEPLOYMENT_INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    
    print(f"\n🎉 Deployment files ready in: {deploy_dir}")
    print("\nNext steps:")
    print("1. Go to https://huggingface.co/new-space")
    print("2. Create a new Gradio Space")
    print("3. Upload the files from hf_deployment/")
    print("4. Upload and extract run_balanced.tar.gz")
    print("5. Wait for deployment to complete")
    print(f"\n📁 All files are in: {deploy_dir.absolute()}")

if __name__ == "__main__":
    prepare_for_hf_spaces()
