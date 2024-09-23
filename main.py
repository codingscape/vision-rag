import sys
import random
import requests
import torch
from diffusers import StableDiffusion3Pipeline
import googlemaps
import os
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, T5EncoderModel, BitsAndBytesConfig # Added T5EncoderModel & BitsAndBytesConfig
import re

HF_API_KEY = None
GOOGLE_API_KEY = None
HEADINGS = [0, 90, 180, 270]
IMAGE_FOLDER = "images"

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
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    for idx, coord in enumerate(coords):
        for heading in HEADINGS:
            url = f"https://maps.googleapis.com/maps/api/streetview?size=600x300&location={coord[0]},{coord[1]}&heading={heading}&key={GOOGLE_API_KEY}"
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
    rand = random.sample(result['results'], int(size / 5))

    coords = []
    for r in rand:
        coords.append((r['geometry']['location']['lat'], r['geometry']['location']['lng']))

    print("Got location coordinates")

    return coords

def generate_image_descriptions():
    print("Generating image descriptions")

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        quantization_config=llava_quantization_config, # set quant
        device_map="cuda:0", # inline set device
        attn_implementation="flash_attention_2", # flash attn 2
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
                 "text": "What is shown in each image? I don't care or want to know about watermarks. Keep it concise. Especially pay attention to weather, architecture, and lighting."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=[prompt], return_tensors="pt").to("cuda:0")

    response = model.generate(**inputs, max_new_tokens=4096)

    decoded = processor.decode(response[0], skip_special_tokens=True)
    cleaned = re.sub(r'\[INST\][\w\s\d\?]*.*\[\/INST\]', '', decoded, flags=re.MULTILINE).strip()

    print("Image descriptions generated")

    print(cleaned)

    return cleaned

def combine_descriptions(descriptions):
    print("Combining descriptions")

    combined_prompt = "Summarize the following descriptions, try to keep specific things about weather, architecture, and lighting: " + " ".join(descriptions)

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        quantization_config=llava_quantization_config, # set quant
        device_map="cuda:0", # inline set device
        attn_implementation="flash_attention_2", # flash attn 2
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": combined_prompt}
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt").to("cuda:0")

    response = model.generate(**inputs, max_new_tokens=400)

    decoded = processor.decode(response[0], skip_special_tokens=True)
    summary_description = re.sub(r'\[INST\][\w\s\d\?]*.*\[\/INST\]', '', decoded, flags=re.MULTILINE).strip()

    print("Descriptions combined")

    return summary_description

def create_optimized_prompt(llava_description):
    optimized_prompt = f"The location looks like this: {llava_description}"
    return optimized_prompt

def generate_final_image(prompt):
    print("Generating final image")

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

    
    image = pipe(prompt).images[0]
    image.save("final_image.png")

    print("Final image generated.")

def main():
    global GOOGLE_API_KEY, HF_API_KEY

    try:
        if '--google' in sys.argv:
            google_index = sys.argv.index('--google')
            GOOGLE_API_KEY = sys.argv[google_index + 1]

        if '--hf' in sys.argv:
            hf_index = sys.argv.index('--hf')
            HF_API_KEY = sys.argv[hf_index + 1]

        if not GOOGLE_API_KEY:
            GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        if not HF_API_KEY:
            HF_API_KEY = os.environ.get('HF_API_KEY')

        if not GOOGLE_API_KEY or not HF_API_KEY:
            raise ValueError

    except (IndexError, ValueError):
        print("Error: Invalid arguments provided")
        sys.exit(2)

    # Step 1: Fetch Street View image
    coords = get_location_coordinates(user_prompt)
    get_street_view_image(coords)

    print()
    print()

    # Step 2: Get image description from LLaVA
    image_descriptions = generate_image_descriptions()
    print(f"Generated description: {image_descriptions}")

    print()
    print()

    # Step 3: Combine descriptions
    summary = combine_descriptions(image_descriptions)
    print(summary)

    print()
    print()

    # Step 4: Create optimized prompt
    optimized_prompt = create_optimized_prompt(summary)
    # print(f"Optimized prompt: {optimized_prompt}")

    print()
    print()

    # Step 5: Generate final image using SDXL-Turbo
    generate_final_image(optimized_prompt)

user_prompt = "a shop in Beverly Hills on Rodeo Drive"
main()
