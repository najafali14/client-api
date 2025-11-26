from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import requests
import base64
import uuid
from rembg import remove
import time
import os

app = FastAPI()

# CORS middleware for Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# environment variable
# Get API keys
GEMINI_API_KEY = "your api key"                # you can also add by environment variable
IMGBB_API = "your api key"  # to get ImgBB Api visit https://api.imgbb.com/ . this is a free images hosting platform that also provide url

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# imgBB API key
IMGBB_API_KEY = IMGBB_API

# ------------------------------
# Helper: Upload image to imgBB
# ------------------------------
def upload_to_imgbb(pil_image):
    try:
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
            },
            timeout=30
        )

        data = response.json()
        if data.get("success"):
            return data["data"]["url"]
        else:
            raise ValueError(f"imgBB upload failed: {data}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

# ------------------------------
# Background Removal (rembg)
# ------------------------------
def remove_background(pil_image):
    try:
        # PIL image ko bytes mein convert karo
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # rembg use karke background remove karo
        output_bytes = remove(img_bytes.getvalue())
        
        # Output bytes ko PIL image mein convert karo
        output_image = Image.open(BytesIO(output_bytes))
        return output_image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

# ------------------------------
# Helper: Generate image with Gemini
# ------------------------------
def generate_image(prompt, input_image):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, input_image]
        )
        
        # Updated response handling
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            img_bytes = part.inline_data.data
                            img = Image.open(BytesIO(img_bytes))
                            return img
        return None
    except Exception as e:
        print(f"Gemini generation error: {str(e)}")
        return None

# ------------------------------
@app.get("/")
def api_home():
    return {
        "message": "Welcome to Shopify Image Enhancement API",
        "status": "active",
        "timestamp": int(time.time())
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "shopify-image-enhancer"}

# ------------------------------
# API Route - FIXED: No external prompts, only image
# ------------------------------
@app.post("/api/generate")
async def generate_image_api(file: UploadFile = File(...)):
    """
    Shopify compatible endpoint - only accepts image file
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file
        file_bytes = await file.read()
        
        # Validate file size
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 10MB allowed.")

        input_image = Image.open(BytesIO(file_bytes))
        
        # Validate image dimensions
        if input_image.size[0] < 100 or input_image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image too small. Minimum 100x100 pixels required.")

        # FIXED: Internal prompts - Shopify se koi prompt nahi milega
        prompt1 = "Create a clean Pixar-style cartoon illustration of the pet in the uploaded photo. Keep the full body, original pose, markings, proportions, and aspect ratio. Use soft shading and vibrant but natural colors. Remove the entire background and output a fully transparent PNG with clean edges."
        
        prompt2 = "Transform the uploaded pet photo into a bright Pixar-inspired character. Keep all body parts visible, maintain the pose, markings, and original aspect ratio. Use glossy highlights and 3D depth while staying natural. Return a transparent PNG with a perfect cut-out."

        # Generate 2 images
        img1 = generate_image(prompt1, input_image)
        img2 = generate_image(prompt2, input_image)

        # If Gemini fails, use original images as fallback
        if not img1 and not img2:
            # Both failed, use original
            img1 = input_image
            img2 = input_image
        elif not img1:
            # First failed, use second for both
            img1 = img2
        elif not img2:
            # Second failed, use first for both
            img2 = img1

        response_data = {
            "success": True,
            "original_filename": file.filename,
            "original_size": f"{input_image.width}x{input_image.height}",
            "timestamp": int(time.time()),
            "images": []
        }

        for i, img in enumerate([img1, img2], start=1):
            try:
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

                response_data["images"].append({
                    "variation": i,
                    "preview_url": preview_url,
                    "highres_url": highres_url,
                    "preview_size": f"{preview_img.width}x{preview_img.height}",
                    "highres_size": f"{highres_img.width}x{highres_img.height}"
                })

            except Exception as e:
                print(f"Error processing variation {i}: {str(e)}")
                continue

        if not response_data["images"]:
            raise HTTPException(status_code=500, detail="All image processing failed")

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Test endpoint for basic functionality
@app.post("/api/test")
async def test_endpoint(file: UploadFile = File(...)):
    """Simple test endpoint"""
    try:
        file_bytes = await file.read()
        image = Image.open(BytesIO(file_bytes))
        return {
            "success": True,
            "message": "API is working",
            "filename": file.filename,
            "size": f"{image.width}x{image.height}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
