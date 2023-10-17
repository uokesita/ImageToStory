from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import torch
import requests
import os
import re
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# image to text

def image_2_text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)

    # print(text)
    return text

# text to speech
def generate_story(scenario):
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.float16, device_map="auto")

    messages = [
        {
            "role": "system",
            "content": "You are a story teller; You can generate a short story based on a single narrative, the story should be no more than 30 words long",
        },
        {"role": "user", "content": str(scenario)},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(substring_after(outputs[0]["generated_text"], "<|assistant|>"))

    return substring_after(outputs[0]["generated_text"], "<|assistant|>")

def substring_after(s, delim):
    return s.partition(delim)[2]

def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="Use AI to create a story from an image")
    st.header("Use AI to create a story from an image")

    uploaded_file = st.file_uploader("Choose an image", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = image_2_text(uploaded_file.name)
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")

# scenario = image_2_text('photo.jpg')
# story = generate_story(scenario)
# text_to_speech(story)

if __name__ == '__main__':
    main()