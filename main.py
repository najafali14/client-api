from fastapi import FastAPI, UploadFile, File, HTTPException, Form  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from google import genai
from google.genai import types  # type: ignore
from PIL import Image
from io import BytesIO
import requests
import base64
import uuid
import json
from rembg import remove  # rembg import karo
from typing import Optional

app = FastAPI()

# CORS middleware for Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.myshopify.com", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client
client = genai.Client(api_key="AIzaSyDZHArLMbWTNAoTtcxUyt5-Q2BuK8vTZro")

# imgBB API key
IMGBB_API_KEY = "8bc5018a821c22732b70c15045f6f903"

# ------------------------------
# Helper: Upload image to imgBB
# ------------------------------
def upload_to_imgbb(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": IMGBB_API_KEY,
            "image": img_base64,
            "name": f"generated_{uuid.uuid4()}"
        }
    )

    data = response.json()
    if data.get("success"):
        return data["data"]["url"]
    else:
        raise ValueError(f"imgBB upload failed: {data}")

# ------------------------------
# Background Removal (rembg)
# ------------------------------
def remove_background(pil_image):
    # PIL image ko bytes mein convert karo
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    # rembg use karke background remove karo
    output_bytes = remove(img_bytes.getvalue())
    
    # Output bytes ko PIL image mein convert karo
    output_image = Image.open(BytesIO(output_bytes))
    return output_image

# ------------------------------
# Helper: Generate image with Gemini
# ------------------------------
def generate_image(prompt, input_image):
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, input_image]
    )
    for part in response.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            img_bytes = part.inline_data.data
            img = Image.open(BytesIO(img_bytes))
            return img
    return None

# ------------------------------
# Helper: Download image from URL
# ------------------------------
def download_image_from_url(image_url: str):
    response = requests.get(image_url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

# ------------------------------
# Helper: Process base64 image
# ------------------------------
def process_base64_image(base64_string: str):
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

# ------------------------------
@app.get("/")
def api_home():
    return "Welcome to Shopify Image Enhancement API"

# ------------------------------
# Shopify-compatible API Routes
# ------------------------------

# Option 1: File Upload (FastAPI UploadFile)
@app.post("/api/generate/upload")
async def generate_image_upload(
    file: UploadFile = File(...),
    product_id: Optional[str] = Form(None),
    variant_id: Optional[str] = Form(None)
):
    try:
        # Read uploaded file
        file_bytes = await file.read()
        input_image = Image.open(BytesIO(file_bytes))

        return await process_and_generate_images(input_image, product_id, variant_id, file.filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Option 2: Image URL
@app.post("/api/generate/url")
async def generate_image_url(
    image_url: str = Form(...),
    product_id: Optional[str] = Form(None),
    variant_id: Optional[str] = Form(None)
):
    try:
        # Download image from URL
        input_image = download_image_from_url(image_url)
        return await process_and_generate_images(input_image, product_id, variant_id, "url_image")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL processing failed: {str(e)}")

# Option 3: Base64 Image
@app.post("/api/generate/base64")
async def generate_image_base64(
    base64_image: str = Form(...),
    product_id: Optional[str] = Form(None),
    variant_id: Optional[str] = Form(None)
):
    try:
        # Process base64 image
        input_image = process_base64_image(base64_image)
        return await process_and_generate_images(input_image, product_id, variant_id, "base64_image")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 processing failed: {str(e)}")

# ------------------------------
# Main Processing Function
# ------------------------------
async def process_and_generate_images(input_image, product_id, variant_id, filename):
    prompt1 = """
    Create a clean Pixar-style cartoon illustration of the pet in the uploaded photo.
Keep the full body, original pose, markings, proportions, and aspect ratio.
Use soft shading and vibrant but natural colors.
Remove the entire background and output a fully transparent PNG with clean edges.
    """
    
    prompt2 = """
    Transform the uploaded pet photo into a bright Pixar-inspired character.
Keep all body parts visible, maintain the pose, markings, and original aspect ratio.
Use glossy highlights and 3D depth while staying natural.
Return a transparent PNG with a perfect cut-out.
    """

    # Generate 2 images
    img1 = generate_image(prompt1, input_image)
    img2 = generate_image(prompt2, input_image)

    if not img1 or not img2:
        raise HTTPException(status_code=500, detail="Image generation failed")

    response_data = {
        "success": True,
        "product_id": product_id,
        "variant_id": variant_id,
        "original_filename": filename,
        "generated_images": []
    }

    for i, img in enumerate([img1, img2], start=1):
        # Remove background using rembg
        img_no_bg = remove_background(img)

        # Preview (small) - long side 768px
        preview_img = img_no_bg.copy()
        width, height = preview_img.size
        if width > height:
            new_width = 768
            new_height = int((768 / width) * height)
        else:
            new_height = 768
            new_width = int((768 / height) * width)
        preview_img = preview_img.resize((new_width, new_height))

        # High-resolution - long side 2400px
        highres_img = img_no_bg.copy()
        width, height = highres_img.size
        if width > height:
            new_width = 2400
            new_height = int((2400 / width) * height)
        else:
            new_height = 2400
            new_width = int((2400 / height) * width)
        highres_img = highres_img.resize((new_width, new_height))

        # Upload to imgBB
        preview_url = upload_to_imgbb(preview_img)
        highres_url = upload_to_imgbb(highres_img)

        response_data["generated_images"].append({
            "variation": i,
            "preview_url": preview_url,
            "highres_url": highres_url,
            "preview_size": f"{preview_img.width}x{preview_img.height}",
            "highres_size": f"{highres_img.width}x{highres_img.height}"
        })

    return JSONResponse(content=response_data)

# ------------------------------
# Health check for Shopify
# ------------------------------
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "shopify-image-enhancer"}
