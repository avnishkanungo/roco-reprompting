import os
import openai
import json

assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
OPENAI_KEY = json.load(open("openai_key.json"))
openai.api_key = str(OPENAI_KEY["OPENAI_KEY"])

print(openai.api_key)