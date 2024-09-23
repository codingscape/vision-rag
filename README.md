# Codingscape Vision RAG Image Generator
This application uses real life imagery to generate images that match the user's description.

## How it works
1. Find places using Google's Place API
2. Send coordinates from step 1 to Google Street view and get images from 0, 90, 180, and 270 degrees for up to 4 of the places
3. Send those images to llava to generate an overall description of the images
4. Use condensed description as a prompt to stable diffusion

## Model versions
- llava-hf/llava-v1.6-mistral-7b-hf
- stabilityai/stable-diffusion-3-medium-diffusers

# Running

## Deployment
SSH is the only supported way of running this on anything that's not local. `scp` `main.py` and `requirements.txt` to your chosen environment.

## Application
Pytorch must be installed separately due to varying hardware differences. Once that is installed, run `pip install -r requirements.txt`

Right now, it's a standalone script with the prompt hard coded into it. There are two ways to pass your Google and Hugging Face keys. Either CLI args or ENV vars.
CLI args are `--google` and `--hf`. The ENV vars are `GOOGLE_API_KEY` and `HF_API_KEY`. You must use one of the two. Edit the prompt and run `python main.py`.

You will need some pretty beefy hardware to run this on unless you have access to RunPod or GPU instances in AWS.