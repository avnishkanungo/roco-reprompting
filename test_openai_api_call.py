import openai
import json
import os

assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
OPENAI_KEY = json.load(open("openai_key.json"))
openai.api_key = str(OPENAI_KEY["OPENAI_KEY"]) ##### Uncomment after LLMA Test

response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "user", "content": "Write a limerick about the wonders of GPU computing."},                                    
                    ],
                    max_tokens=1024,
                    temperature=0.2,
                    )
usage = response['usage']
response = response['choices'][0]['message']["content"]
print('======= response ======= \n ', response)
print('======= usage ======= \n ', usage)


