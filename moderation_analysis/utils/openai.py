

import os
import json
import tqdm
import time

from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_key = os.getenv("OPENAI_ORG_KEY")

client = OpenAI(api_key=openai_api_key, organization=openai_org_key)

# The number of max trial for api calls before abandoning or raie error
MAX_TRIALS = 3

# Default function for whow prompt to validate the gpt generated output
def check_single_answer(output_text, labels=None):
    try:
        answer = json.loads(output_text)
        if not labels or len(labels) == 5:
            if any([k not in answer for k in ["motives", "dialogue act", "target speaker(s)"] ]):
                return False
            else:
                motives = answer["motives"]
                if motives:
                    for m in motives:
                        if m not in ["informational motive", "social motive", "coordinative motive"]:
                            return False
                dialogue_act = answer["dialogue act"]
                if dialogue_act not in ["Probing", "Confronting", "Supplement", "Interpretation", "Instruction", "All Utility"]:
                    return False
        else:
            if labels[0] == "dialogue act":
                dialogue_act = answer["dialogue act"]
                if dialogue_act not in ["Probing", "Confronting", "Supplement", "Interpretation", "Instruction",
                                        "All Utility"]:
                    return False
            elif "motive" in labels[0]:
                if "verdict" not in answer :
                    return False
                pred = int(answer["verdict"])
                if pred != 1 and pred != 0:
                    return False
            else:
                if "target speaker(s)" not in answer :
                    return False
                pred = int(answer["target speaker(s)"].split(" ")[0])
                if pred > 12 or pred < 0:
                    return False
        return True
    except Exception as e:
        return False


def gpt35(input_text, model="gpt-3.5-turbo-0125"):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1,
        response_format={"type": "json_object"}
    )
    output_text = completion.choices[0].message.content
    return output_text

def gpt4(input_text, model="gpt-4o"):
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1,
        response_format={"type": "json_object"}
    )
    output_text = completion.choices[0].message.content
    return output_text


def gpt_single(input_text, api_func, tries=3, wait_time=1):
    output_text = None
    for n in range(tries + 1):
        if n == tries:
            raise Exception(f"Tried {tries} times.")
        try:
            output_text = api_func(input_text)
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text


def gpt_batch(batch_prompts, model_name='gpt-40', labels=None, valid_func=None):
    results = []
    api_func = None
    output = None
    if model_name == 'gpt35':
        api_func = gpt35
    elif model_name == 'gpt-4o':
        api_func = gpt4
    for i, instance in tqdm(enumerate(batch_prompts)):
        trial = 0
        is_ans_validate = False
        while trial < MAX_TRIALS:
            output = gpt_single(instance["prompt"], api_func=api_func, tries=3, wait_time=1)
            if valid_func:
                if valid_func(output):
                    is_ans_validate = True
                    break
            else:
                if check_single_answer(output, labels=labels):
                    is_ans_validate = True
                    break

            trial += 1
        if is_ans_validate:
            instance["output"] = json.loads(output)
        else:
            instance["output"] = None
        results.append(instance)
    return results