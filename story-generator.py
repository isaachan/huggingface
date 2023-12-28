import dotenv
dotenv.load_dotenv()

from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import requests
import sys

def img_to_text(url):
    p = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=40)
    text = p(url)[0]["generated_text"]
    return text

def generate_story(scenario):
    template = """
    你是一位很会讲故事的人，下面的Context是一段英文，请你根据这句话讲个故事，要求用中文，并且带点儿幽默诙谐的那种。字数控制在100字以内。
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1), 
                         prompt=prompt,
                         verbose=True)

    story = story_llm.predict(scenario=scenario)
    return story

def text_to_speech(story):
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = { "inputs": story }
    repsonse = requests.post(API_URL, headers=headers, json=payload)
    if repsonse.status_code == 200:
        with open("audio.flac", "wb") as f:
            f.write(repsonse.content)
        print("Audio saved to audio.flac")
    else:
        print("Something went wrong")
        print(repsonse.text)

if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    url = input("Please input the image url: ")

text = img_to_text(url)
print(text)

print("Here is the story: ")
story = generate_story(text)
print(story)

text_to_speech(story)