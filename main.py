import sys
import random
import requests
import torch
from diffusers import StableDiffusion3Pipeline
import googlemaps
import os
from PIL import Image
from huggingface_hub import HfFolder
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, T5EncoderModel, BitsAndBytesConfig # Added T5EncoderModel & BitsAndBytesConfig
import re
import shutil
import glob
import math

HF_API_KEY = ""
GOOGLE_API_KEY = ""
HEADINGS = [0, 90, 180, 270]
IMAGE_FOLDER = "images"
MAX_PROMPT_LENGTH = 500
NUM_IMAGES = 0
PREFERRED_LOCATIONS = 4
USE_FLASH_ATTN=True

# llava quantization config
llava_quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
# sd quantization config
sd_quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def get_street_view_image(coords):
    print("Fetching Street View images")

    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)

    os.makedirs(IMAGE_FOLDER)

    for idx, coord in enumerate(coords):
        for heading in HEADINGS:
            url = f"https://maps.googleapis.com/maps/api/streetview?size=600x400&location={coord[0]},{coord[1]}&heading={heading}&key={GOOGLE_API_KEY}&fov=100"
            response = requests.get(url)

            if response.status_code == 200:
                image_path = os.path.join(IMAGE_FOLDER, f'street_view_heading_{idx}_{heading}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Image saved at {image_path}")
            else:
                print(f"Error fetching image for heading {heading}. Status code: {response.status_code}")

    print("Street View images fetched")

def get_location_coordinates(description):
    print("Getting location coordinates")

    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

    result = gmaps.places(description, "textquery")
    size = len(result['results'])
    ratio = size / PREFERRED_LOCATIONS

    rand = random.sample(result['results'], math.ceil(size / ratio))

    coords = []
    for r in rand:
        coords.append((r['geometry']['location']['lat'], r['geometry']['location']['lng']))

    print("Got location coordinates")

    return coords

def generate_image_descriptions():
    print("Generating image descriptions")

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    kwargs = {"torch_dtype":torch.float16, "quantization_config":llava_quantization_config, "device_map":"cuda:0"}

    if USE_FLASH_ATTN:
        kwargs.update({"attn_implementation":"flash_attention_2"})

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        **kwargs
    )

    images = []
    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            image = Image.open(image_path)

            images.append(image)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "What is shown in the image? I don't care or want to know about watermarks. Be as descriptive as possible. Especially pay attention to weather, architecture, and lighting."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to("cuda:0")

    response = model.generate(**inputs, max_new_tokens=4096)

    decoded = processor.decode(response[0], skip_special_tokens=True)
    cleaned = re.sub(r'\[INST\][\w\s\d\?]*.*\[\/INST\]', '', decoded, flags=re.MULTILINE).strip()

    print("Image descriptions generated")

    print(cleaned)

    return cleaned

def create_optimized_prompt(llava_description):
    optimized_prompt = f"The location looks like this: {llava_description}"
    return optimized_prompt

def generate_final_image(prompt, index):
    print("Generating final image")

    HfFolder.save_token(HF_API_KEY)

    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_3",
        quantization_config=sd_quantization_config,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=text_encoder,
        device_map="balanced",
        torch_dtype=torch.float16
    )

    # The main prompt needs to be shortened to avoiding warnings about the model clipping it
    split = prompt.split()

    image = pipe(
        prompt=" ".join(split[0:50]),
        prompt_3=" ".join(split[0:MAX_PROMPT_LENGTH]),
        negative_prompt="letters and numbers that aren't real, cars facing different directions in the same lane",
        guidance_scale=10,
        num_inference_steps=30,
        max_sequence_length=MAX_PROMPT_LENGTH
    ).images[0]

    image.save(f"final_image_{index}.png")

    print(f"Image {index + 1} of {NUM_IMAGES} generated.")

def main():
    global GOOGLE_API_KEY, HF_API_KEY, NUM_IMAGES, USE_FLASH_ATTN

    prompt = ""

    try:
        if '--google' in sys.argv:
            google_index = sys.argv.index('--google')
            GOOGLE_API_KEY = sys.argv[google_index + 1]

        if '--hf' in sys.argv:
            hf_index = sys.argv.index('--hf')
            HF_API_KEY = sys.argv[hf_index + 1]

        if '--num' in sys.argv:
            amount_index = sys.argv.index('--num')
            NUM_IMAGES = int(sys.argv[amount_index + 1])

        if '--prompt' in sys.argv:
            prompt_index = sys.argv.index('--prompt')
            prompt = sys.argv[prompt_index + 1]

        if '--no-flash-attn' in sys.argv:
            USE_FLASH_ATTN = False

        if not GOOGLE_API_KEY:
            GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        if not HF_API_KEY:
            HF_API_KEY = os.environ.get('HF_API_KEY')
        if not NUM_IMAGES:
            NUM_IMAGES = int(os.environ.get('NUM_IMAGES', 1))

        if not GOOGLE_API_KEY or not HF_API_KEY or not prompt:
            raise ValueError

    except (IndexError, ValueError):
        print("Error: Invalid arguments provided")
        sys.exit(2)

    files = glob.glob('*.png')
    for path in files:
        try:
            os.remove(path)
        except OSError:
            print("Error while deleting file")

    for i in range(NUM_IMAGES):
        # Step 1: Fetch Street View image
        coords = get_location_coordinates(prompt)
        get_street_view_image(coords)

        print()
        print()

        # Step 2: Get image description from LLaVA
        image_descriptions = generate_image_descriptions()
        print(f"Generated description: {image_descriptions}")

        print()
        print()

        # Step 3: Create optimized prompt
        optimized_prompt = create_optimized_prompt(image_descriptions)

        print()
        print()

        # Step 4: Generate final image using SDXL-Turbo
        generate_final_image(optimized_prompt, i)

main()
