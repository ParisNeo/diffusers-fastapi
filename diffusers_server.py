"""
Project: diffusers-fastapi

A FastAPI-based server for image generation, image-to-image transformation, and inpainting using Hugging Face Diffusers.

Author: ParisNeo
GitHub: https://github.com/ParisNeo/diffusers-fastapi

This project provides a robust API for various image generation and manipulation tasks using state-of-the-art
diffusion models. It supports text-to-image generation, image-to-image transformation, and inpainting, making
it a versatile tool for AI-powered image creation and editing.

Features:
- Text-to-image generation
- Image-to-image transformation
- Inpainting
- Support for multiple Stable Diffusion models
- Automatic package management and updates
- Configurable output and model caching directories
- Verbose logging option for debugging

Usage:
Run the script with the desired arguments to start the FastAPI server. Use the provided endpoints to interact
with the image generation and manipulation capabilities.

Example:
python diffusers_fastapi.py --host 0.0.0.0 --port 8253 --model "stabilityai/stable-diffusion-2-1" --outputs_dir "output" --models_dir "models" --verbose

License: Apache 2.0
"""

import argparse
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline, AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from typing import Optional
import uvicorn
import logging
import pipmaster as pm
import pkg_resources
from PIL import Image
import io

global diffusers_model
outputs_folder = Path("__file__").parent/"outputs"
outputs_folder.mkdir(exist_ok=True, parents=True)
models_folder = Path("__file__").parent/"models"
models_folder.mkdir(exist_ok=True, parents=True)

app = FastAPI(
    title="Diffusers FastAPI",
    description="A FastAPI server for image generation and manipulation using Hugging Face Diffusers",
    version="1.0.0",
)

class ImageGenerationRequest(BaseModel):
    """
    Pydantic model for image generation requests.
    """
    positive_prompt: str
    negative_prompt: str = ""
    sampler_name: str = ""
    seed: int = -1
    scale: float = 7.5
    steps: int = 20
    width: int = 512
    height: int = 512
    restore_faces: bool = True

class ImageGenerationResponse(BaseModel):
    """
    Pydantic model for image generation responses.
    """
    image_path: str
    prompt: str
    negative_prompt: str

class DiffusersModel:
    """
    A class to manage different Stable Diffusion models and perform various image generation tasks.
    """

    def __init__(self, diffusers_model: str, outputs_dir: str, models_dir: str, verbose: bool = False):
        """
        Initialize the DiffusersModel.

        Args:
            diffusers_model (str): The name or path of the Diffusers model to use.
            outputs_dir (str): Directory to save generated images.
            models_dir (str): Directory to cache models.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
        """
        self.model = None
        self.img2img_model = None
        self.inpaint_model = None
        self.outputs_dir = outputs_dir
        self.models_dir = models_dir
        self.diffusers_model = diffusers_model
        self.verbose = verbose
        self.load_model()

    def load_model(self):
        """
        Load the Stable Diffusion models and required packages.
        """
        if self.verbose:
            logging.info(f"Loading model: {self.diffusers_model}")
        
        required_packages = [
            ["torch", "", "https://download.pytorch.org/whl/cu121"],
            ["diffusers", "0.30.1", None],
            ["transformers", "4.44.2", None],
            ["accelerate", "0.33.0", None],
            ["imageio-ffmpeg", "0.5.1", None]
        ]

        for package, min_version, index_url in required_packages:
            if not pm.is_installed(package):
                if self.verbose:
                    logging.info(f"Installing {package}")
                pm.install_or_update(package, index_url)
            else:
                if min_version:
                    if pkg_resources.parse_version(pm.get_installed_version(package)) < pkg_resources.parse_version(min_version):
                        if self.verbose:
                            logging.info(f"Updating {package} to version {min_version}")
                        pm.install_or_update(package, index_url)
        
        try:
            use_cuda = torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
            torch_dtype = torch.float16 if use_cuda else torch.float32

            if self.verbose:
                logging.info(f"Using device: {device}")
                logging.info(f"Using dtype: {torch_dtype}")

            model_kwargs = {
                "cache_dir": self.models_dir,
                "use_safetensors": True,
                "torch_dtype": torch_dtype,
            }

            if "stable-diffusion-3" in self.diffusers_model:
                self.model = StableDiffusion3Pipeline.from_pretrained(
                    self.diffusers_model, 
                    **model_kwargs
                )
            else:
                self.model = AutoPipelineForText2Image.from_pretrained(
                    self.diffusers_model, 
                    **model_kwargs
                )
            self.model.to(device)
            try:
                # Load img2img model
                self.img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.diffusers_model,
                    **model_kwargs
                )
                self.img2img_model.to(device)

                # Load inpainting model
                self.inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
                    self.diffusers_model,
                    **model_kwargs
                )
                self.inpaint_model.to(device)
            except:
                self.img2img_model = None
                self.inpaint_model = None

            if self.verbose:
                logging.info("Models loaded successfully")
        except Exception as e:
            if self.verbose:
                logging.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def paint(self, request: ImageGenerationRequest) -> tuple:
        """
        Generate an image based on the given request.

        Args:
            request (ImageGenerationRequest): The image generation request.

        Returns:
            tuple: A tuple containing the path to the generated image and metadata.
        """
        if self.verbose:
            logging.info(f"Generating image with prompt: {request.positive_prompt}")
        if request.sampler_name:
            sc = self.get_scheduler_by_name(request.sampler_name)
            if sc:
                self.model.scheduler = sc
        
        width = adjust_dimensions(request.width)
        height = adjust_dimensions(request.height)
        
        generator = torch.Generator("cuda").manual_seed(request.seed) if request.seed != -1 else None
        
        try:
            image = self.model(
                request.positive_prompt,
                negative_prompt=request.negative_prompt,
                height=height,
                width=width,
                guidance_scale=request.scale,
                num_inference_steps=request.steps,
                generator=generator
            ).images[0]
        except Exception as e:
            if self.verbose:
                logging.error(f"Image generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

        output_path = Path(self.outputs_dir)
        fn = find_next_available_filename(output_path, "diff_img_")
        image.save(fn)
        
        if self.verbose:
            logging.info(f"Image saved to: {fn}")
        return str(fn), {"prompt": request.positive_prompt, "negative_prompt": request.negative_prompt}

    def img2img(self, init_image: Image.Image, request: ImageGenerationRequest) -> tuple:
        """
        Perform image-to-image generation based on the given request and initial image.

        Args:
            init_image (Image.Image): The initial image to transform.
            request (ImageGenerationRequest): The image generation request.

        Returns:
            tuple: A tuple containing the path to the generated image and metadata.
        """
        if self.verbose:
            logging.info(f"Generating image-to-image with prompt: {request.positive_prompt}")
        
        generator = torch.Generator("cuda").manual_seed(request.seed) if request.seed != -1 else None
        
        try:
            image = self.img2img_model(
                prompt=request.positive_prompt,
                image=init_image,
                negative_prompt=request.negative_prompt,
                guidance_scale=request.scale,
                num_inference_steps=request.steps,
                generator=generator
            ).images[0]
        except Exception as e:
            if self.verbose:
                logging.error(f"Image-to-image generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image-to-image generation failed: {str(e)}")

        output_path = Path(self.outputs_dir)
        fn = find_next_available_filename(output_path, "img2img_")
        image.save(fn)
        
        if self.verbose:
            logging.info(f"Image-to-image result saved to: {fn}")
        return str(fn), {"prompt": request.positive_prompt, "negative_prompt": request.negative_prompt}

    def inpaint(self, init_image: Image.Image, mask_image: Image.Image, request: ImageGenerationRequest) -> tuple:
        """
        Perform inpainting based on the given request, initial image, and mask.

        Args:
            init_image (Image.Image): The initial image to inpaint.
            mask_image (Image.Image): The mask indicating areas to inpaint.
            request (ImageGenerationRequest): The image generation request.

        Returns:
            tuple: A tuple containing the path to the generated image and metadata.
        """
        if self.verbose:
            logging.info(f"Generating inpainting with prompt: {request.positive_prompt}")
        
        generator = torch.Generator("cuda").manual_seed(request.seed) if request.seed != -1 else None
        
        try:
            image = self.inpaint_model(
                prompt=request.positive_prompt,
                image=init_image,
                mask_image=mask_image,
                negative_prompt=request.negative_prompt,
                guidance_scale=request.scale,
                num_inference_steps=request.steps,
                generator=generator
            ).images[0]
        except Exception as e:
            if self.verbose:
                logging.error(f"Inpainting failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inpainting failed: {str(e)}")

        output_path = Path(self.outputs_dir)
        fn = find_next_available_filename(output_path, "inpaint_")
        image.save(fn)
        
        if self.verbose:
            logging.info(f"Inpainting result saved to: {fn}")
        return str(fn), {"prompt": request.positive_prompt, "negative_prompt": request.negative_prompt}

diffusers_model = None

@app.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Endpoint for text-to-image generation.

    Args:
        request (ImageGenerationRequest): The image generation request.

    Returns:
        ImageGenerationResponse: The response containing the generated image path and metadata.
    """
    image_path, metadata = diffusers_model.paint(request)
    return ImageGenerationResponse(
        image_path=image_path,
        prompt=metadata["prompt"],
        negative_prompt=metadata["negative_prompt"]
    )

@app.post("/img2img", response_model=ImageGenerationResponse)
async def img2img(
    image: UploadFile = File(...),
    positive_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: int = Form(-1),
    scale: float = Form(7.5),
    steps: int = Form(20)
):
    """
    Endpoint for image-to-image generation.

    Args:
        image (UploadFile): The initial image file.
        positive_prompt (str): The positive prompt for image generation.
        negative_prompt (str, optional): The negative prompt for image generation. Defaults to "".
        seed (int, optional): The random seed for generation. Defaults to -1.
        scale (float, optional): The guidance scale. Defaults to 7.5.
        steps (int, optional): The number of inference steps. Defaults to 20.

    Returns:
        ImageGenerationResponse: The response containing the generated image path and metadata.
    """
    contents = await image.read()
    init_image = Image.open(io.BytesIO(contents)).convert("RGB")
    request = ImageGenerationRequest(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        scale=scale,
        steps=steps
    )
    image_path, metadata = diffusers_model.img2img(init_image, request)
    return ImageGenerationResponse(
        image_path=image_path,
        prompt=metadata["prompt"],
        negative_prompt=metadata["negative_prompt"]
    )

@app.post("/inpaint", response_model=ImageGenerationResponse)
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    positive_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: int = Form(-1),
    scale: float = Form(7.5),
    steps: int = Form(20)
):
    """
    Endpoint for inpainting.

    Args:
        image (UploadFile): The initial image file.
        mask (UploadFile): The mask image file.
        positive_prompt (str): The positive prompt for image generation.
        negative_prompt (str, optional): The negative prompt for image generation. Defaults to "".
        seed (int, optional): The random seed for generation. Defaults to -1.
        scale (float, optional): The guidance scale. Defaults to 7.5.
        steps (int, optional): The number of inference steps. Defaults to 20.

    Returns:
        ImageGenerationResponse: The response containing the generated image path and metadata.
    """
    image_contents = await image.read()
    mask_contents = await mask.read()
    init_image = Image.open(io.BytesIO(image_contents)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_contents)).convert("RGB")
    request = ImageGenerationRequest(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        scale=scale,
        steps=steps
    )
    image_path, metadata = diffusers_model.inpaint(init_image, mask_image, request)
    return ImageGenerationResponse(
        image_path=image_path,
        prompt=metadata["prompt"],
        negative_prompt=metadata["negative_prompt"]
    )

def adjust_dimensions(dim: int) -> int:
    """
    Adjust dimensions to be divisible by 8 (required by some models).

    Args:
        dim (int): The input dimension.

    Returns:
        int: The adjusted dimension.
    """
    return (dim // 8) * 8

def find_next_available_filename(path: Path, prefix: str) -> Path:
    """
    Find the next available filename in the given path with the specified prefix.

    Args:
        path (Path): The directory path.
        prefix (str): The filename prefix.

    Returns:
        Path: The next available filename.
    """
    i = 0
    while True:
        file_path = path / f"{prefix}{i}.png"
        if not file_path.exists():
            return file_path
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusers Image Generation API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8253, help="Port to run the server on")
    parser.add_argument("--model", type=str, default="v2ray/stable-diffusion-3-medium-diffusers", help="Diffusers model to use")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save generated images")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to cache models")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    diffusers_model = DiffusersModel(args.model, args.outputs_dir, args.models_dir, verbose=args.verbose)

    if args.verbose:
        logging.info(f"Starting server on {args.host}:{args.port}")
        logging.info(f"Using model: {args.model}")
        logging.info(f"Output directory: {args.outputs_dir}")
        logging.info(f"Models directory: {args.models_dir}")

    uvicorn.run(app, host=args.host, port=args.port)
