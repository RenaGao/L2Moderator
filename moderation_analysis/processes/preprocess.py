import glob
import json
from nltk import sent_tokenize
import pandas as  pd


from moderation_analysis.meta_info import topics, moderators
from moderation_analysis.utils.data import merge_short_strings


# Process the segmented and messy transcript strings from the same author to better organized utterances.
def merge_and_process_utterance_cache(utterance_cache, turn_index, speaker):

    merged_utterance = []

    # merged all utterance belong to the same speaker to a block.
    text_block = " ".join([u["text"] for u in utterance_cache])

    # breakdown the block to multiple sentences
    sentences = sent_tokenize(text_block)

    # merge short sentences(e.g. "Yea.") into longer utterance.
    sentences = merge_short_strings(sentences)
    for j, sent in enumerate(sentences):
        merged_utterance.append({
            "turn_index": turn_index,
            "sent_index": j,
            "segment": 0,
            "speaker": speaker,
            "role": utterance_cache[-1]["role"],
            "text": sent
        })

    return merged_utterance


def preprocess_episode(eps_file):
    df = pd.read_csv(eps_file, index_col=0)
    session_id = eps_file.split("/")[-1].replace(".csv", "")
    topic  = topics[session_id.split("_")[0]]
    if "Without" in session_id:
        moderator = None
    else:
        moderator = moderators[session_id]
    # record participant ids
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

            merged_turns = merge_and_process_utterance_cache(utterance_cache, turn_index, last_speaker)
            new_eps.extend(merged_turns)

            # update turn index
            turn_index += 1

            # clear utterance cache since speaker changes.
            utterance_cache = []

        # adding each utterance from the same speaker to the cache for merge and process later.
        utterance_cache.append({
                "speaker": speaker,
                "role": role,
                "text": content
            })
        last_speaker = speaker

    if utterance_cache:
        merged_turns = merge_and_process_utterance_cache(utterance_cache, turn_index, last_speaker)
        new_eps.extend(merged_turns)

    # collect speaker meta for annotation options
    speakers_options = ["Unknown", "Everyone"] + participants
    speakers_options = [f"{i} ({sp})" for i, sp in enumerate(speakers_options)]

    # create meta dict for the episode
    meta = {"moderator": moderator, "session_id": session_id, "topic": topic, "speakers":speakers_options}
    return pd.DataFrame(new_eps), meta

def preprocess(source_folder, output_folder, overwrite=False):
    session_files = glob.glob(source_folder + "*.csv")
    for f in session_files:
        df, meta = preprocess_episode(f)
        file_name = output_folder + meta["session_id"] + "_pro.csv"
        df.to_csv(file_name)
        with open(output_folder + meta["session_id"] +"_pro_meta.json", "w") as outfile:
            json.dump(meta, outfile)



if __name__ == '__main__':
    preprocess("../../data/raw/", "../../data/unannotated/")