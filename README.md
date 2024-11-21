# Implementing RoCo: Dialectic Multi-Robot Collaboration with Large Language Models with reprompting


Based on the paper: https://arxiv.org/pdf/2307.04738

## Setup
### setup conda env and package install
```
conda create -n roco python=3.8 
conda activate roco
```
### Install mujoco and dm_control 
```
pip install mujoco==2.3.0
pip install dm_control==1.0.8 
```
**If you have M1 Macbook like me and would like to visualize the task scenes locally:**

Download the macos-compatible `.dmg` file from [MuJoCo release page](https://github.com/deepmind/mujoco/releases), inside it should have a `MuJoCo.app` file that you can drag into your /Application folder, so it becomes just like other apps in your Mac. You could then open up the app and drag xml files in it. Find more informationa in the [official documentation](https://mujoco.readthedocs.io/en/latest/programming/#getting-started).

### Install other packages
```
pip install -r requirements.txt
```

### Acquire OpenAI/Claude API Keys
This is required for prompting GPTs or Claude LLMs. You don't necessarily need both of them. Put your key string somewhere safely in your local repo, and provide a file path (something like `./roco/openai_key.json`) and load them in the scripts. Example code snippet:
```
import openai  
openai.api_key = YOUR_OPENAI_KEY

import anthropic
client = anthropic.Client(api_key=YOUR_CLAUDE_KEY)
streamed = client.completion_stream(...)  
```

## Usage 
### Run multi-robot dialog on the PackGrocery Task using the latest GPT-4 model
```
$ conda activate roco
(roco) $ python run_dialog.py --task pack -llm gpt-4
```
## Current Updates
Updated OpenAI Library so that we can use the current standard functions for API call used for chat completion. This allows the usage of the Llama 3.1 model finetuned and made available by NVIDIA.
