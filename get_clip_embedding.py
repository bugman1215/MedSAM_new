import torch
import torch.nn as nn
from PIL import Image
from open_clip import create_model_from_pretrained  # Adjust this import according to your actual CLIP library
import torchvision.transforms as T


def get_clip_embeddings(image) -> torch.Tensor:
    clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    embed_dim = 256

    clip_model, clip_preprocess = create_model_from_pretrained(clip_model_name)

    if isinstance(image, torch.Tensor):
        if image.ndim == 4:  # Handle the batch dimension
            image = image[0]  # Take the first image in the batch

    image = T.ToPILImage()(image)

    clip_inputs = clip_preprocess(image).unsqueeze(0)



    with torch.no_grad():
        clip_outputs = clip_model.visual(clip_inputs)
        clip_image_embeddings = clip_outputs
        clip_prompt_embeddings = nn.Linear(512, embed_dim)(clip_image_embeddings)
        clip_prompt_embeddings = clip_prompt_embeddings.view(1, 1, embed_dim)

    return clip_prompt_embeddings

