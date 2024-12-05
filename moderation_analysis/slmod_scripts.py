import pandas as pd
import glob
from whow_prompts import construct_prompt_unit as construct_whow_prompt
from slmod_act_prompts import construct_prompt_unit as  construct_slmod_prompt, attributes
from nltk import sent_tokenize
import json
from  import gpt_batch
from utils.dataloaders import load_json_data




PRIOR_CONTEXT_SIZE = 5
POST_CONTEXT_SIZE = 2

CORPUS = "slmod"
MODEL = "gpt-4o"






def validate_slmod_answer(output_text):
    try:
        answer = json.loads(output_text)
        if "dialogue act" not in answer or "target speaker(s)" not in answer:
            return False

        dialogue_act = extract_text_in_parentheses(answer["dialogue act"])
        if dialogue_act not in list(attributes["dialogue act"]["options"].keys()):
            return False

        pred_ts_index = int(answer["target speaker(s)"].split(" ")[0])
        if pred_ts_index > 12 or pred_ts_index < 0:
            return False

        return True
    except Exception as e:
        return False

def preprocess_episode(eps_file):
    df = pd.read_csv(eps_file, index_col=0)
    session_id = eps_file.split("/")[-1].replace(".csv", "")
    topic  = topics[session_id.split("_")[0]]
    if "Without" in session_id:
        moderator = None
    else:
        moderator = moderators[session_id]
    participants = []
    new_eps = []

    utterance_cache = []
    last_speaker = None
    turn_index = 0

    for i, r in df.iterrows():
        raw_text = r.Text.split(":")
        speaker = raw_text[0].strip()
        content = raw_text[1].strip()
        role = "participant"
        if speaker == moderator:
            role = "moderator"
        elif speaker not in participants:
            participants.append(speaker)

        # The end of a speaker's turn
        if speaker != last_speaker and utterance_cache:
            text_block = " ".join([u["text"] for u in utterance_cache])
            sentences = sent_tokenize(text_block)
            sentences = merge_short_strings(sentences)
            for j, sent in enumerate(sentences):
                new_eps.append({
                    "turn_index": turn_index,
                    "sent_index": j,
                    "segment": 0,
                    "speaker": last_speaker,
                    "role": utterance_cache[-1]["role"],
                    "text": sent
                })
            turn_index += 1
            utterance_cache = []

        utterance_cache.append({
                "speaker": speaker,
                "role": role,
                "text": content
            })
        last_speaker = speaker

    if utterance_cache:
        text_block = " ".join([u["text"] for u in utterance_cache])
        sentences = sent_tokenize(text_block)
        sentences = merge_short_strings(sentences)
        for j, sent in enumerate(sentences):
            new_eps.append({
                "turn_index": turn_index,
                "sent_index": j,
                "segment": 0,
                "speaker": last_speaker,
                "role": utterance_cache[-1]["role"],
                "text": sent
            })

    speakers_options = ["Unknown", "Everyone"] + participants
    speakers_options = [f"{i} ({sp})" for i, sp in enumerate(speakers_options)]
    meta = {"moderator": moderator, "session_id": session_id, "topic": topic, "speakers":speakers_options}
    return pd.DataFrame(new_eps), meta

def generate_batch_prompt(eps, meta, labels, slmod_acts=True):

    gpt_prompts = []
    topic = meta["topic"]
    session_id = meta["session_id"]
    for i, r in eps.iterrows():
        if r.role == "mod":
            utt_id = int(r.id.split("_")[0])
            sent_id = int(r.id.split("_")[1])
            prior_context_mask = eps.id.apply(lambda x: not (
                    not (utt_id - PRIOR_CONTEXT_SIZE <= int(x.split("_")[0]) < utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) < sent_id)))
            post_context_mask = eps.id.apply(lambda x: not (
                    not (utt_id + POST_CONTEXT_SIZE >= int(x.split("_")[0]) > utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) > sent_id)))
            context = {
                "prior_context":[],
                "post_context": []
            }

            if len(eps[prior_context_mask]) > 0:
                context["prior_context"] = [(v.speaker, v.role, v.text) for i, v in eps[prior_context_mask].iterrows()]

            if len(eps[post_context_mask ]) > 0:
                context["post_context"] = [(v.speaker, v.role, v.text) for i, v in eps[post_context_mask].iterrows()]
            instance = {
                "id": r.id,
                "meta": meta,
                "context": context,
                "target": (r.speaker, r.role, r.text)
            }
            if slmod_acts:
                prompt = construct_slmod_prompt(instance)
            else:
                prompt = construct_whow_prompt(instance, CORPUS, labels)
            instance["prompt"] = prompt
            gpt_call = {"custom_id": session_id.split("/")[-1].replace(".xlsx", "") + "_" + r.id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": MODEL,
                                 "messages": [{"role": "user", "content": prompt}],
                                 "temperature": 1,
                                 "response_format": {"type": "json_object"},
                                 "max_tokens": 300}
                        }

            gpt_prompts.append(gpt_call)
    return gpt_prompts

def load_slmod_episodes():
    session_files = glob.glob("./processed/*.csv")
    session_files = [f for f in session_files if "pro" in f and "WithMod" in f]
    sessions_meta = glob.glob("./processed/*_meta.json")
    sessions_meta = [f for f in sessions_meta  if "WithMod" in f]
    result_files = glob.glob("./processed/*_result.json")
    return session_files, sessions_meta, result_files

def process_episode(conv, conv_meta, slmod_acts=True):

    batch_tasks = []
    ids = pd.Series(conv['turn_index'].astype(str) + '_' + conv['sent_index'].astype(str))
    for i, r in conv.iterrows():
        if r.role == "moderator":
            utt_id = r.turn_index
            sent_id = r.sent_index
            id = str(utt_id) + "_" + str(sent_id)

            prior_context_mask = ids.apply(lambda x: not (
                    not (utt_id - PRIOR_CONTEXT_SIZE <= int(x.split("_")[0]) < utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) < sent_id)))
            post_context_mask = ids.apply(lambda x: not (
                    not (utt_id + POST_CONTEXT_SIZE >= int(x.split("_")[0]) > utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) > sent_id)))
            context = {
                "prior_context":[],
                "post_context": []
            }

            if len(conv[prior_context_mask]) > 0:
                context["prior_context"] = [(v.speaker, v.role, v.text) for i, v in conv[prior_context_mask].iterrows()]

            if len(conv[post_context_mask ]) > 0:
                context["post_context"] = [(v.speaker, v.role, v.text) for i, v in conv[post_context_mask].iterrows()]
            instance = {
                "id": id,
                "meta": conv_meta,
                "context": context,
                "target": (r.speaker, r.role, r.text)
            }
            if slmod_acts:
                prompt = construct_slmod_prompt(instance)
            else:
                prompt = construct_whow_prompt(instance, CORPUS)
            instance["prompt"] = prompt
            batch_tasks.append(instance)
    return batch_tasks


def preprocess():
    session_files = glob.glob("./*.csv")
    for f in session_files:
        if "pro" in f: continue
        df, meta = preprocess_episode(f)
        file_name = "processed/" + meta["session_id"] + "_pro.csv"
        df.to_csv(file_name)
        with open("processed/" + meta["session_id"] +"_pro_meta.json", "w") as outfile:
            json.dump(meta, outfile)

def annotate(slmod_acts=True):
    session_files, sessions_metas, result_files = load_slmod_episodes()
    for sf in session_files:
        print(sf)
        json_file = sf.replace(".csv", "_meta.json")
        meta_file = [m for m in sessions_metas if m == json_file][0]
        conv = pd.read_csv(sf, index_col=0)
        with open(meta_file) as f:
            conv_meta = json.load(f)

        if slmod_acts:
            result_file = "./processed/" + conv_meta["session_id"] + "_slmod_result.json"
        else:
            result_file = "./processed/" + conv_meta["session_id"] +"_result.json"
        existing_results = [r for r in result_files if result_file == r]
        if len(existing_results) > 0:
            print(f"result exist at {result_file}")
            continue

        task_instances = process_episode(conv, conv_meta)
        results = gpt_batch(task_instances, model_name=MODEL, valid_func=validate_slmod_answer)
        if results:
            with open(result_file, "w") as outfile:
                json.dump(results, outfile)

def integrate_result_to_df(whow_annotation=True):
    result_files = glob.glob("./mod_result/*_slmod_result.json")
    results = {}
    for r in result_files:
        if whow_annotation:
            session_id = r.split("/")[-1].replace("_slmod_result.json", "")
        else:
            session_id = r.split("/")[-1].replace("_slmod_result.json", "")
        result = load_json_data(r)
        result = {i["id"]:i for i in result}
        results[session_id] = result

    metas = {}
    meta_files = glob.glob("./processed/*_meta.json")
    for m in meta_files:
        meta = load_json_data(m)
        session_id = meta["session_id"]
        metas[session_id] = meta

    files = glob.glob("./processed/*.csv")
    for f in files:
        if "WithMod" not in f:
            continue
        df = pd.read_csv(f, index_col=0)
        session_id = f.split("/")[-1].replace("_pro.csv", "")
        result = results[session_id]
        meta = metas[session_id]

        dialogue_act = []
        target_speakers = []
        reasons = []

        if whow_annotation:
            informational = []
            social = []
            coordinative = []


        for i, r in df.iterrows():
            if r.role == "moderator":
                id = str(r.turn_index) + "_" + str(r.sent_index)
                gpt_label = result[id]["output"]
                dialogue_act.append(gpt_label["dialogue act"])
                target_speakers.append(gpt_label["target speaker(s)"])
                reasons.append(gpt_label["reason"])

                if whow_annotation:
                    informational.append("informational motive" in gpt_label["motives"])
                    coordinative.append("coordinative motive" in gpt_label["motives"])
                    social.append("social motive" in gpt_label["motives"])

            else:
                dialogue_act.append(None)
                target_speakers.append(None)
                reasons.append(None)

                if whow_annotation:
                    informational.append(None)
                    coordinative.append(None)
                    social.append(None)

        if whow_annotation:
            df["dialogue_act"] = dialogue_act
            df["informational motive"] = informational
            df["coordinative motive"] = coordinative
            df["social motive"] = social
        else:
            df["slmod_act"] = dialogue_act
            df["target speaker(s)"] = target_speakers
        df["reason"] = reasons

        if whow_annotation:
            df.to_csv(f.replace("processed/", "mod_result/").replace("pro", "whow"))
        else:
            df.to_csv(f.replace("processed/", "mod_result/").replace("pro", "slmod"))



if __name__ == '__main__':
    # preprocess()
    # annotate()
    integrate_result_to_df(whow_annotation=False)