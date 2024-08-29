# Diffusers FastAPI
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)](https://fastapi.tiangolo.com/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

Diffusers FastAPI is a robust and versatile FastAPI-based server for image generation, image-to-image transformation, and inpainting using Hugging Face Diffusers. This project provides a powerful API for various image generation and manipulation tasks using state-of-the-art diffusion models.

## Features

- Text-to-image generation
- Image-to-image transformation
- Inpainting
- Support for multiple Stable Diffusion models
- Automatic package management and updates
- Configurable output and model caching directories
- Verbose logging option for debugging

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

To start the Diffusers FastAPI server, run the following command:

```
python diffusers_fastapi.py [options]
```

### Options

- `--host`: Host to run the server on (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 8253)
- `--model`: Diffusers model to use (default: "v2ray/stable-diffusion-3-medium-diffusers")
- `--output_dir`: Directory to save generated images (default: "output")
- `--models_dir`: Directory to cache models (default: "models")
- `--verbose`: Enable verbose logging

Example:
```
python diffusers_fastapi.py --host 0.0.0.0 --port 8253 --model "stabilityai/stable-diffusion-2-1" --output_dir "output" --models_dir "models" --verbose
```

## API Endpoints

The server provides the following endpoints:

1. `/generate-image`: Text-to-image generation
2. `/img2img`: Image-to-image transformation
3. `/inpaint`: Inpainting

For detailed information about request and response formats, please refer to the API documentation available at `/docs` when the server is running.

## Python Examples

Here are some examples of how to use the API with Python requests:

### Text-to-Image Generation

```python
import requests
import json

url = "http://localhost:8253/generate-image"
payload = {
    "positive_prompt": "A beautiful landscape with mountains and a lake",
    "negative_prompt": "clouds, people",
    "seed": 42,
    "scale": 7.5,
    "steps": 20,
    "width": 512,
    "height": 512
}

response = requests.post(url, json=payload)
result = json.loads(response.text)
print(f"Generated image path: {result['image_path']}")
```

### Image-to-Image Transformation

```python
import requests

url = "http://localhost:8253/img2img"
files = {
    'image': ('input.png', open('input.png', 'rb'), 'image/png')
}
data = {
    'positive_prompt': 'Transform this landscape into a snowy scene',
    'negative_prompt': 'summer, green',
    'seed': 42,
    'scale': 7.5,
    'steps': 20
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Generated image path: {result['image_path']}")
```

### Inpainting

```python
import requests

url = "http://localhost:8253/inpaint"
files = {
    'image': ('input.png', open('input.png', 'rb'), 'image/png'),
    'mask': ('mask.png', open('mask.png', 'rb'), 'image/png')
}
data = {
    'positive_prompt': 'Add a cat sitting on the couch',
    'negative_prompt': 'dog, bird',
    'seed': 42,
    'scale': 7.5,
    'steps': 20
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Generated image path: {result['image_path']}")
```

## API Documentation

For detailed API documentation, including request and response schemas, please run the server and navigate to `http://localhost:8253/docs` in your web browser. This will open the Swagger UI, which provides interactive documentation for all available endpoints.

## License

This project is licensed under the Apache 2.0 License.

## Author

ParisNeo - A computer geek passionate about AI

GitHub: [https://github.com/ParisNeo](https://github.com/ParisNeo)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.