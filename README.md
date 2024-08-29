# Diffusers FastAPI

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)](https://fastapi.tiangolo.com/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

A FastAPI-based service for image generation using Hugging Face's Diffusers library.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ParisNeo/diffusers-fastapi.git
   cd diffusers-fastapi
   ```

2. Install PyTorch:
   For Windows with CUDA 12.1:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   For other systems, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

3. Install other dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the FastAPI server:

```
python diffusers_api.py --host 0.0.0.0 --port 8080 --model "stabilityai/stable-diffusion-2-1" --output_dir "./output" --models_dir "./models"
```

You can customize the host, port, model, and directories by modifying the command-line arguments.

## API Endpoints

- `POST /generate-image`: Generate an image based on the provided parameters.

For detailed API documentation, visit `http://localhost:8080/docs` after starting the server.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

ParisNeo

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [FastAPI](https://fastapi.tiangolo.com/)