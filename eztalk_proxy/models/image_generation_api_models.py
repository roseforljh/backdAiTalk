from pydantic import BaseModel, Field
from typing import List, Optional

class ImageGenerationRequest(BaseModel):
    model: str = Field(..., description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    image_size: Optional[str] = Field("1024x1024", description="The size of the generated images.")
    batch_size: Optional[int] = Field(1, description="The number of images to generate.")
    num_inference_steps: Optional[int] = Field(20, description="The number of denoising steps.")
    guidance_scale: Optional[float] = Field(7.5, description="Higher guidance scale encourages to generate images that are closely linked to the text prompt.")
    apiAddress: str = Field(..., description="The address of the downstream API service.")
    apiKey: str = Field(..., description="The API key for the downstream service.")

class ImageUrl(BaseModel):
    url: str

class Timings(BaseModel):
    inference: int

class ImageGenerationResponse(BaseModel):
    images: List[ImageUrl]
    timings: Timings
    seed: int