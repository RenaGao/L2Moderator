import glob
import pandas as pd
import json

from moderation_analysis.utils.openai import gpt_batch
from moderation_analysis.utils.data import extract_text_in_parentheses, load_json_data
from moderation_analysis.prompts.slmod_act_prompts import attributes, construct_prompt_unit as construct_slmod_prompt
from moderation_analysis.prompts.whow_prompts import construct_prompt_unit as construct_whow_prompt



MODEL = "gpt-4o"
CORPUS = "slmod"


def process_episode(conv, conv_meta, whow_annotation=True, prior_context_len=5, post_context_len=2):

    batch_tasks = []
    ids = pd.Series(conv['turn_index'].astype(str) + '_' + conv['sent_index'].astype(str))
    for i, r in conv.iterrows():
        if r.role == "moderator":
            utt_id = r.turn_index
            sent_id = r.sent_index
            id = str(utt_id) + "_" + str(sent_id)

            prior_context_mask = ids.apply(lambda x: not (
                    not (utt_id - prior_context_len <= int(x.split("_")[0]) < utt_id) and not (int(x.split("_")[0]) == utt_id and \
                                                                                                int(x.split("_")[1]) < sent_id)))
            post_context_mask = ids.apply(lambda x: not (
                    not (utt_id + post_context_len >= int(x.split("_")[0]) > utt_id) and not (int(x.split("_")[0]) == utt_id and \
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
            if whow_annotation:
                prompt = construct_whow_prompt(instance, CORPUS)
            else:
                prompt = construct_slmod_prompt(instance)
            instance["prompt"] = prompt
            batch_tasks.append(instance)
    return batch_tasks

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

def load_slmod_episodes(source_folder, meta_folder, output_folder):
    session_files = glob.glob(source_folder + "*.csv")
    session_files = [f for f in session_files if "pro" in f and "WithMod" in f]
    sessions_meta = glob.glob(meta_folder + "*_meta.json")
    sessions_meta = [f for f in sessions_meta  if "WithMod" in f]
    result_files = glob.glob(output_folder + "*_result.json")
    return session_files, sessions_meta, result_files


def annotate(source_folder, meta_folder, output_folder, whow_annotation=True, overwrite=False):

    # get source unannotated data, meta data, and existing results.
    session_files, sessions_metas, result_files = load_slmod_episodes(source_folder, meta_folder, output_folder)
    for sf in session_files:
        print(sf)
        json_file = sf.replace(".csv", "_meta.json")

        # find match meta data
        meta_file = [m for m in sessions_metas if m == json_file][0]
        conv = pd.read_csv(sf, index_col=0)
        with open(meta_file) as f:
            conv_meta = json.load(f)

        if whow_annotation:
            result_file = output_folder + conv_meta["session_id"] + "_whow_result.json"
        else:
            result_file = output_folder + conv_meta["session_id"] + "_slmod_result.json"

        # check existing results for the session id.
        existing_results = [r for r in result_files if result_file == r]
        if len(existing_results) > 0 and not overwrite:
            print(f"result exist at {result_file}: skip!")
            continue

        # generate annotation prompts for each moderation sentences
        task_instances = process_episode(conv, conv_meta)

        # annotating using gpt api
        if whow_annotation:
            results = gpt_batch(task_instances, model_name=MODEL)
        else:
            # use different validation function for validating the moderator act output.
            results = gpt_batch(task_instances, model_name=MODEL, valid_func=validate_slmod_answer)

        # save results
        if results:
            with open(result_file, "w") as outfile:
                json.dump(results, outfile)

# This function load the api generated result jsons to the transcript csvs.
def integrate_result_to_df(source_folder, meta_folder, output_folder, whow_annotation=True):
    if whow_annotation:
        result_files = glob.glob(output_folder + "*_whow_result.json")
    else:
        result_files = glob.glob(output_folder + "*_slmod_result.json")

    # load the result jsons.
    results = {}
    for r in result_files:
        if whow_annotation:
            session_id = r.split("/")[-1].replace("_whow_result.json", "")
        else:
            session_id = r.split("/")[-1].replace("_slmod_result.json", "")

        result = load_json_data(r)
        result = {i["id"]:i for i in result}
        results[session_id] = result

    # load meta data.
    metas = {}
    meta_files = glob.glob(meta_folder + "/*_meta.json")
    for m in meta_files:
        meta = load_json_data(m)
        session_id = meta["session_id"]
        metas[session_id] = meta

    # load the source csv
    files = glob.glob(source_folder + "*.csv")

    # iterating through each session transcript csv and add results.
    for f in files:
        if "WithMod" not in f:
            continue
        df = pd.read_csv(f, index_col=0)
        session_id = f.split("/")[-1].replace("_pro.csv", "")
        result = results[session_id]

        dialogue_act = []
        target_speakers = []
        reasons = []

        # whow annotation contains motives
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

                # whow annotation contains motives
                if whow_annotation:
                    informational.append("informational motive" in gpt_label["motives"])
                    coordinative.append("coordinative motive" in gpt_label["motives"])
                    social.append("social motive" in gpt_label["motives"])

            else:
                dialogue_act.append(None)
                target_speakers.append(None)
                reasons.append(None)

                # whow annotation contains motives
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
            # For slmod annotation, we use slmod acts instead of the original dialogue acts
            df["slmod_act"] = dialogue_act
            df["target speaker(s)"] = target_speakers
        df["reason"] = reasons

        if whow_annotation:
            df.to_csv(f.replace(source_folder, output_folder).replace("pro", "whow"))
        else:
            df.to_csv(f.replace(source_folder, output_folder).replace("pro", "slmod"))








if __name__ == '__main__':

    # If whow_annotation True, means using the default whow framework prompt to annotate
    # Else using the developed slmod prompt for annotation
    whow_annotation = False

    source_folder = "../../data/unannotated/"
    meta_folder = "../../data/meta/"

    if whow_annotation:
        output_folder = "../../data/whow_annotated/"
    else:
        output_folder = "../../data/slmod_annotated/"

    annotate(source_folder, meta_folder, output_folder)
    integrate_result_to_df(whow_annotation=False)