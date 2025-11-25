from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
from typing import Optional
import time

app = FastAPI(title="Shopify Image Enhancement API")

# Proper CORS for Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.myshopify.com",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gemini client
client = genai.Client(api_key="AIzaSyDZHArLMbWTNAoTtcxUyt5-Q2BuK8vTZro")
IMGBB_API_KEY = "8bc5018a821c22732b70c15045f6f903"

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
                "name": f"shopify_{uuid.uuid4()}"
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
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        output_bytes = remove(img_bytes.getvalue())
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
            model="gemini-2.0-flash-exp",
            contents=[prompt, input_image]
        )
        
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                img_bytes = part.inline_data.data
                img = Image.open(BytesIO(img_bytes))
                return img
        return None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

# ------------------------------
# Shopify-compatible API Routes
# ------------------------------
@app.get("/")
def api_home():
    return {
        "message": "Shopify Image Enhancement API", 
        "status": "active",
        "version": "1.0"
    }

@app.post("/api/shopify/generate")
async def shopify_generate_image(
    file: UploadFile = File(...),
    shop_domain: Optional[str] = Form(None),
    product_id: Optional[str] = Form(None),
    variant_id: Optional[str] = Form(None),
    customer_id: Optional[str] = Form(None)
):
    """
    Shopify-compatible image generation endpoint
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size (max 10MB)
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 10MB allowed.")
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        input_image = Image.open(BytesIO(file_bytes))
        
        # Validate image dimensions
        if input_image.size[0] < 100 or input_image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image too small. Minimum 100x100 pixels required.")

        # Prompts for Shopify
        prompt1 = """Create a clean Pixar-style cartoon illustration of the pet. Keep full body, original pose, markings, proportions. Use soft shading and vibrant colors. Remove background for transparent PNG."""
        
        prompt2 = """Transform the pet into Pixar-inspired character. Maintain pose and markings. Use glossy highlights and 3D depth. Return transparent PNG."""

        # Generate images
        img1 = generate_image(prompt1, input_image)
        img2 = generate_image(prompt2, input_image)

        if not img1 and not img2:
            raise HTTPException(status_code=500, detail="Image generation failed")

        response_data = {
            "success": True,
            "shop_domain": shop_domain,
            "product_id": product_id,
            "variant_id": variant_id,
            "customer_id": customer_id,
            "timestamp": int(time.time()),
            "generated_images": []
        }

        images = [img for img in [img1, img2] if img is not None]
        
        for i, img in enumerate(images, start=1):
            try:
                # Remove background
                img_no_bg = remove_background(img)

                # Preview image
                preview_img = img_no_bg.copy()
                width, height = preview_img.size
                if width > height:
                    new_width = 768
                    new_height = int((768 / width) * height)
                else:
                    new_height = 768
                    new_width = int((768 / height) * width)
                preview_img = preview_img.resize((new_width, new_height))

                # High-res image
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

            except Exception as e:
                # Continue with other images if one fails
                print(f"Error processing variation {i}: {e}")
                continue

        if not response_data["generated_images"]:
            raise HTTPException(status_code=500, detail="All image processing failed")

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Health check for Shopify app
@app.get("/api/shopify/health")
async def shopify_health_check():
    return {
        "status": "healthy",
        "service": "shopify-image-enhancer",
        "timestamp": int(time.time())
    }

# Webhook endpoint for Shopify
@app.post("/api/shopify/webhooks/order-created")
async def shopify_webhook(payload: dict):
    """
    Handle Shopify webhooks
    """
    return {"status": "webhook_received", "webhook_type": "order_created"}
