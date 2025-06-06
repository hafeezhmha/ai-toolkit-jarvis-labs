# Training Flux on Jarvis Labs using AI Toolkit by Ostris

## Installation

**Requirements:**
- Python >=3.10
- Jarvis Labs Instance
- Python venv or conda env
- Hugging Face account

**Linux:**
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
# For Windows, use: .\venv\Scripts\activate
# Install PyTorch first
pip3 install torch
pip3 install -r requirements.txt
```

**Windows:**
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## FLUX.1 Training

### Requirements

You will need a Jarvis Labs Instance. You can get one at [https://jarvislabs.ai](https://jarvislabs.ai).
Navigate to the template section and select "PyTorch" for a PyTorch instance. Ensure it has **at least 48 GB of VRAM** to train FLUX.
> Recommended: Start with an instance offering 48 GB VRAM (e.g., 1 x A6000) and a storage configuration of at least 130 GB.

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. This means anything you train with it will inherit this non-commercial license. It is also a gated model, so you must accept the license on Hugging Face before using it. Otherwise, the process will fail.

Here are the required steps to set up the license:

1.  Sign into Hugging Face and accept the model access terms here: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2.  Create a file named `.env` in the root directory of this project.
3.  [Get a READ access token from Hugging Face](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so: `HF_TOKEN=your_key_here`

### Training

1.  Copy an example configuration file from `config/examples/train_lora_flux_24gb.yaml` (or `config/examples/train_lora_flux_schnell_24gb.yaml` for Schnell) to the `config` folder. Rename it (e.g., `my_training_config.yml`).
2.  Edit the copied configuration file, following the comments within the file.
3.  Run the training script using your configuration file: `python run.py config/my_training_config.yml`
4.  An example configuration used to train a FLUX model on bedrooms is available at `config/examples/train_hk_lora.yaml` for reference.

A folder will be created with the name specified in your configuration file. This folder will contain all checkpoints and generated images. You can stop the training at any time using `Ctrl+C`. When you resume, it will pick up from the last saved checkpoint.

**IMPORTANT:** If you press `Ctrl+C` while a checkpoint is being saved, it will likely corrupt that checkpoint. Wait until the saving process is complete.

## Gradio UI

To get started with the training UI, once you have followed the steps above and `ai-toolkit` is installed:

```bash
cd ai-toolkit # If you are not already in the ai-toolkit folder
huggingface-cli login # Provide a 'write' token to publish your LoRA at the end
python flux_train_ui.py
```

This will launch a UI that allows you to upload your images, caption them using Florence, train your LoRA, and publish it. This is recommended if you have not yet captioned your images.

## Training on Jarvis Labs

#### Example Configuration ($0.5/hr):

-   1x A6000 (48 GB VRAM)
-   7 vCPU, 32 GB RAM

> **Custom Storage Configuration:** (For saving datasets, samples, and generated models)
> - 130 GB

### 1. Setup

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
# Optional: Run if you encounter issues
pip install --upgrade accelerate transformers diffusers huggingface_hub
```

### 2. Upload Your Dataset

-   Create a new folder in the project's root directory (e.g., `dataset`).
-   Drag and drop your `.jpg`, `.jpeg`, or `.png` images and their corresponding `.txt` caption files into this newly created dataset folder.

### 3. Log in to Hugging Face with an Access Token

-   Get a READ token from [here](https://huggingface.co/settings/tokens).
-   Request access to the FLUX.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
-   Run `huggingface-cli login` in your terminal and paste your token when prompted.

### 4. Training

-   Copy an example configuration file from `config/examples` to the `config` folder and rename it (e.g., `my_jarvis_training.yml`).
-   Edit the configuration file, following the comments.
-   Change `folder_path: "/path/to/images/folder"` to your dataset path, for example: `folder_path: "/workspace/ai-toolkit/your-dataset"`.
-   Run the training script:
    ```bash
    python run.py config/my_jarvis_training.yml
    ```
-   Once training is completed, you can directly export your model to Hugging Face using the script: `python scripts/hf_model.py`.

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the supported image formats are `.jpg`, `.jpeg`, and `.png`. WebP format currently has issues.

The text files should be named the same as their corresponding images but with a `.txt` extension (e.g., `image2.jpg` and `image2.txt`). The text file should contain only the caption for the image. You can include the placeholder `[trigger]` in your caption files; if `trigger_word` is defined in your configuration, `[trigger]` will be automatically replaced with its value.

Images are never upscaled. They are downscaled and placed into buckets for batching.
One of the best features of AI Toolkit is that **you do not need to crop or resize your images manually**. The data loader will automatically resize them and can handle varying aspect ratios.

Happy Training!
