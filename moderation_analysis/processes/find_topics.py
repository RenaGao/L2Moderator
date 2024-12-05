import pandas as pd
import numpy as np
import os
import glob
import json
import re
from bertopic import BERTopic
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy_transformers
import string
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.cluster import KMeans

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import random
import os

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# Load the English language model
nlp = spacy.load('en_core_web_trf')
stopwords = set(stopwords.words('english'))
# Set TOKENIZERS_PARALLELISM to "false" or "true" as needed
os.environ["TOKENIZERS_PARALLELISM"] = "true"

da_encoder = {'Instruction': 2, 'Probing': 0, 'Supplement': 4, 'Interpretation': 3,
              'All Utility': 5, 'Confronting': 1}
dialogue_acts_decode = {0: "probing", 1: "confronting", 2: "instruction", 3: "interpretation",
                        4: "supplement",
                        5: "utility"}

index = ['probing', 'confronting', 'instruction', 'interpretation', 'supplement', 'utility']
col = ['informational', 'coordinative', 'social']
motive_decode = {"i": "informational", "s": "social", "c": "coordinative"}
da_decode = {'prob': "Probing", 'conf': "Confronting", 'inst': "Instruction", 'inte': "Interpretation",
             'supp': 'Supplement', 'util': 'All Utility'}

topic_words = ["romance", "romantic", "relationship", "modern", "society", "boss", "bosses", "employment", "work", "job",
            "stress", "cultural", "ai", "future", "god", "ghost", "superstitions", "superstition"]


def load_results(source_result_folder):
    output = {}
    result_files = glob.glob(source_result_folder + "*.csv")
    for r in result_files:
        session_id = r.split("/")[-1].replace("_pro.csv", "")
        df = pd.read_csv(r, index_col=0)
        output[session_id] = df
    return output


def find_intercepted_interval_indexes(range_tuple, intervals):
    i, j = range_tuple
    intercepted_indexes = []

    for index, interval in enumerate(intervals):
        start, end = interval
        # Check if intervals intersect
        if start <= j and end >= i:
            intercepted_indexes.append(index)

    return intercepted_indexes


def generate_intervals(n, m):
    interval_size = n // m
    intervals = []
    for i in range(m):
        start = i * interval_size
        end = (i + 1) * interval_size - 1
        if i == m - 1:  # Adjust the last interval to include the remainder
            end = n - 1
        intervals.append((start, end))
    return intervals


# aggregate the whow motives and dialogue act labels into inter-utterance forms and intra-utterance forms for analysis.
def aggregate_utterance_labels(results):
    inter_utterance_sequence = []
    intra_utterance_sequence = []
    for session_id, rdf in results.items():
        episode_inter_utterance_sequence = []
        episode_intra_utterance_sequence = []
        total_tokens_count = np.sum([len(str(s).split(" ")) for s in rdf["text"].dropna().tolist()])
        token_count_intervals = generate_intervals(total_tokens_count, 40)

        prior_mod_cache = {
            "speaker": "",
            "role": "",
            "index": -1
        }
        accumulate_tokens_count = 0

        turn_ids = list(rdf.turn_index.unique())

        for i, t_id in enumerate(turn_ids):
            sents = rdf[rdf.turn_index == t_id]
            first_sent = sents.iloc[0]
            speaker = first_sent.speaker
            role = first_sent.role
            text = " ".join(sents.text)
            tokens_count = len(text.split(" "))

            interval_index = find_intercepted_interval_indexes(
                (accumulate_tokens_count, accumulate_tokens_count + tokens_count), token_count_intervals)

            accumulate_tokens_count += tokens_count

            if role == "moderator":
                das = "-".join(list(sents["dialogue_act"].unique()))
                motives = "".join(
                    sorted([m[0] for m in ["informational", "social", "coordinative"] if any(sents[m + " motive"])]))

                cur_utt_info = {"title": session_id, "speaker": speaker, "text": text, "sentence_count": len(sents),
                                "role": role, "interval_index": interval_index, "count": tokens_count,
                                "labels": {"da": das, "m": motives}}

                episode_intra_utterance_sequence.append([{"title": session_id, "speaker": speaker, "text": s.text,
                                                          "role": role, "labels": {"da": da_encoder[s.dialogue_act],
                                                                                   "m": "".join([m[0] for m in
                                                                                                 ["informational",
                                                                                                  "social",
                                                                                                  "coordinative"] if
                                                                                                 s[m + " motive"]])}}
                                                         for i, s in sents.iterrows()])

            else:
                label = "rott."
                if role == prior_mod_cache["role"]:
                    if speaker == prior_mod_cache["speaker"]:
                        label = "cont."

                cur_utt_info = {"title": session_id, "speaker": speaker, "text": text, "sentence_count": len(sents),
                                "role": role, "interval_index": interval_index, "count": tokens_count,
                                "labels": {"transition": label}}

                prior_mod_cache["speaker"] = speaker
                prior_mod_cache["role"] = role
                prior_mod_cache["index"] = i

            episode_inter_utterance_sequence.append(cur_utt_info)

        inter_utterance_sequence.append(episode_inter_utterance_sequence)
        intra_utterance_sequence.append(episode_intra_utterance_sequence)
    return inter_utterance_sequence, intra_utterance_sequence


# Get the motive x dialogue co-occurence matrix
def get_motive_dialogue_act_matrix(label_sequence, index=None, col=None, add_total=True, normalise=True):
    items = []
    labels_count = {"da": [0, 0, 0, 0, 0, 0], "m": [0, 0, 0], "total": 0}
    for sents in label_sequence:
        for sent in sents:
            if "transition" in sent["labels"]:
                continue
            labels_count["total"] += 1
            dialogue_act, motives = sent["labels"]["da"], sent["labels"]["m"]
            try:
                labels_count["da"][int(dialogue_act)] += 1
            except Exception as e:
                dialogue_act = str(int(float(dialogue_act)))
                labels_count["da"][int(dialogue_act)] += 1
            if motives:
                for m in motives:
                    m_index = "ics".index(m)
                    labels_count["m"][m_index] += 1
                    items.append((dialogue_acts_decode[dialogue_act], motive_decode[m]))

    if not index:
        first_dim = [item[0] for item in items]
        unique_first_dim = sorted(set(first_dim))
        index = unique_first_dim

    if not col:
        second_dim = [item[1] for item in items]
        unique_second_dim = sorted(set(second_dim))
        col = unique_second_dim

    # Step 2: Count Co-occurrences
    co_occurrence_counts = pd.DataFrame(0, index=index, columns=col)

    for item in items:
        co_occurrence_counts.at[item[0], item[1]] += 1

    co_occurrence_counts = co_occurrence_counts[[motive_decode['i'], motive_decode['c'], motive_decode['s']]]
    co_occurrence_counts.columns = ["IM", "CM", "SM"]
    co_occurrence_counts = co_occurrence_counts.T
    if normalise:
        co_occurrence_counts = co_occurrence_counts.div(labels_count['m'], axis=0)
        co_occurrence_counts = co_occurrence_counts.round(4)

    co_occurrence_counts = co_occurrence_counts.fillna(0)
    # co_occurrence_counts["total"] = [c / labels_count['total'] for c in labels_count['da']]
    if add_total:
        if normalise:
            co_occurrence_counts["total"] = [c / labels_count['total'] for c in labels_count['m']]
            dialogue_act_count = [c / labels_count['total'] for c in labels_count['da']]
        else:
            co_occurrence_counts["total"] = [c for c in labels_count['m']]
            dialogue_act_count = [c for c in labels_count['da']]
        last_row = pd.DataFrame([dialogue_act_count + [labels_count['total']]], columns=co_occurrence_counts.columns)
        last_row.index = ["total"]
        co_occurrence_counts = pd.concat([co_occurrence_counts, last_row])

    co_occurrence_counts.columns = [re.sub('[^A-Za-z]+', '', c)[:4] if c != "total" else "total" for c in
                                    co_occurrence_counts.columns]

    return co_occurrence_counts, labels_count


def get_motive_dialogue_act_matrix_episode_breakdown(label_sequence, index=None, col=None, normalise=True):
    eps_sqs_dict = {}
    eps_dfs = []
    eps_label_counts = {}
    for seq in label_sequence:
        eps = seq[0][0]["title"]
        eps_sqs_dict[eps] = seq

    for eps, seqs in eps_sqs_dict.items():
        co_occurrence_counts, labels_count = get_motive_dialogue_act_matrix(seqs, index, col, add_total=True,
                                                                            normalise=normalise)
        if co_occurrence_counts.isnull().values.any():
            print("DataFrame contains NaN values.")
        eps_dfs.append(co_occurrence_counts)
        eps_label_counts[eps] = labels_count

    # Stack the DataFrames into a 3D array (n, rows, cols)
    array_3d = np.array([df.values for df in eps_dfs])

    if normalise:
        # Calculate the means and stds for each cell
        mean_array = np.mean(array_3d, axis=0)
        # std_array = np.std(array_3d, axis=0)

        # Create DataFrames from the results
        mean_df = pd.DataFrame(mean_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)
        # std_df = pd.DataFrame(std_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)

        return mean_df, eps_label_counts
    else:
        # Calculate the means and stds for each cell
        sum_array = np.sum(array_3d, axis=0)
        # std_array = np.std(array_3d, axis=0)

        # Create DataFrames from the results
        sum_df = pd.DataFrame(sum_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)
        # std_df = pd.DataFrame(std_array, columns=eps_dfs[0].columns, index=eps_dfs[0].index)
        return sum_df, eps_label_counts


def get_topic_coherence_score(topic_model, sentences):
    # Get topics and their corresponding words
    topics = topic_model.get_topics()
    texts = [doc.split() for doc in sentences]

    # Create a dictionary and corpus for coherence calculation
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Get topic words
    topic_words = [[word for word, _ in topic_model.get_topic(topic_num)] for topic_num in
                   topic_model.get_topic_info().Topic.unique() if topic_num != -1]

    # Calculate coherence
    coherence_model = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence


def get_speaker_strings(results):
    speakers_strings = set()
    for t, df in results.items():
        df_speakers = df.speaker.unique()
        df_speakers = df_speakers.tolist()
        for n in df_speakers:
            name_strings = n.lower().split(" ")
            speakers_strings.update(name_strings)
    speakers_strings = list(speakers_strings)
    speakers_strings.extend(["felicity", "felicitys", "felicity", "moderator", "social", "'s"])
    return speakers_strings


def get_shorten_index(text):
    shorten_index = len(text.split(" "))

    index_comma = 0
    indexes_comma = [i for i, c in enumerate(text) if c == ","]
    for i in indexes_comma:
        if len(text[:i].split(" ")) >= 10:
            index_comma = i
            break

    before_co_str = text[:index_comma]
    before_co_tk_count = len(before_co_str.split(" "))

    index_stop = 0
    indexes_stop = [i for i, c in enumerate(text) if c == "."]
    for i in indexes_stop:
        if len(text[:i].split(" ")) >= 10:
            index_stop = i
            break

    before_stp_str = text[:index_stop]
    before_stp_tk_count = len(before_stp_str.split(" "))

    max_tk_count = max(before_co_tk_count, before_stp_tk_count)
    min_tk_count = min(before_co_tk_count, before_stp_tk_count)

    if max_tk_count < 10:
        shorten_index = 10
    elif min_tk_count >= 10:
        shorten_index = min_tk_count
    else:
        for c in [before_co_tk_count, before_stp_tk_count]:
            if c >= 10:
                shorten_index = c

    return " ".join(text.split(" ")[:shorten_index])


def extract_key_reason(text):
    doc = nlp(text)
    root_toks = []
    for token in doc:
        if token.dep_ == "ROOT":
            root_toks.append(token)

    sents = []
    for rtk in root_toks:
        sent_toks = []
        sent_toks.append(rtk)
        for child in rtk.children:
            if child.dep_ == "dobj":
                for t in child.subtree:
                    sent_toks.append(t)

            # if child.dep_ in ["nsubj", "dobj"]:
            #     sent_toks.append(child)
            # for t in child.subtree:
            #    sent_toks.append(t)

        sent_toks = sorted(sent_toks, key=lambda token: token.i)
        sents.append(" ".join([t.text for t in sent_toks]))

    return sents


def print_top_freq_words(sentences):
    # Initialize an empty list to hold all tokens
    all_tokens = []

    for sentence in sentences:
        # Tokenize the sentence
        tokens = sentence.split(" ")
        # Convert tokens to lowerc
        all_tokens.extend(tokens)

    # Count the frequency of each unique token
    token_counts = Counter(all_tokens)

    # Get the top N tokens
    top_n = 20
    most_common_tokens = token_counts.most_common(top_n)

    # Display the results
    print(f"Top {top_n} unique tokens:")
    for token, count in most_common_tokens:
        print(f"{token}: {count}")


def remove_person_names(text):
    doc = nlp(text)
    # Replace each person name with an empty string
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            text = text.replace(ent.text, '')
    return text


def preprocess(sentences, speakers_strings):

    new_sentences = []
    for text in sentences:
        text = get_shorten_index(text)
        # text = extract_key_reason(text)[0]
        text = remove_person_names(text)
        tokens = text.split(" ")
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t not in speakers_strings]
        tokens = [t for t in tokens if t not in topic_words]
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [t for t in tokens if t not in stopwords]

        text = " ".join(tokens)
        new_sentences.append(text)
    return new_sentences


def show_topics_example(topic_model, topics, sents):
    topic_info = topic_model.get_topic_info()

    topics_examples = {}

    for i, t in enumerate(topics):
        if t not in topics_examples:
            topics_examples[t] = []
        topics_examples[t].append(sents[i])

    for i, r in topic_info.iterrows():
        print(f"Topic index {r.Topic}")
        print(f"Topic count {r.Count}")
        print(f"Topic name {r.Name}")
        print(f"Rep words: {r.Representation}")
        print("Rep sents: \n")
        # topics_example[r.Topic] = []
        examples = random.sample(topics_examples[r.Topic], 4)

        for s in examples:
            print(s)
            print("\n")

        print("*" * 60)


def model_topics(sentences, umap_n_neighbor, umap_min_dist, n_clusters):
    umap_model = UMAP(n_neighbors=umap_n_neighbor, n_components=5, min_dist=umap_min_dist, metric='cosine')
    cluster_model = KMeans(n_clusters=n_clusters)
    topic_model = BERTopic(umap_model=umap_model, embedding_model="allenai-specter",
                           hdbscan_model=cluster_model, calculate_probabilities=True)
    topics, probabilities = topic_model.fit_transform(sentences)
    # Calculate coherence or another metric
    print(f"n_clusters={n_clusters}, umap_n_neighbor={umap_n_neighbor}; umap_min_dists={umap_min_dist}")
    coherence = get_topic_coherence_score(topic_model, sentences)
    print(f"Coherence={coherence}")
    return topic_model, topics, coherence

# grid search for optimized topic modelling params
def find_best_topics(sentences, umap_n_neighbors=[5, 10, 30],
                     umap_min_dists=[0.0, 0.3, 0.5],
                     n_clusterses=[2, 3, 4, 5]):
    best_score = -1
    best_params = {}

    for umap_n_neighbor in umap_n_neighbors:
        for umap_min_dist in umap_min_dists:
            for n_clusters in n_clusterses:
                incomplete = True
                while incomplete:
                    try:
                        topic_model, topics, coherence = model_topics(sentences, umap_n_neighbor, umap_min_dist,
                                                                      n_clusters)
                        incomplete = False
                    except Exception as e:
                        print(e)
                # Update best score and parameters
                if coherence > best_score:
                    best_score = coherence
                    best_params = {
                        'n_clusters': n_clusters,
                        'umap_min_dist': umap_min_dist,
                        'umap_n_neighbor': umap_n_neighbor
                    }
    return best_params

# identify cross labels (e.g. social motive + probing) that occur more frequently than threshold.
def identify_prominent_action_motive_pairs(mean_df, sum_df, rdf, maintain_threshold=0.025, topic_model_threshold=0.1 ):
    remained_cross_labels = {}
    topic_modeling_target = []

    for i in mean_df.index:
        for c in mean_df.columns:
            if c == "total" or i == "total":
                continue
            real_prob = round(mean_df.at[i, c] * mean_df.at[i, 'total'], 3)

            # Only remain the intersected label higher than the maintain_threshold
            if real_prob >= maintain_threshold:
                print("\n")
                print(f"index: {i}, col: {c}")
                print(f"count: {round(sum_df.at[i, c], 3)}")
                print(f"transition prob: {round(mean_df.at[i, c], 3)}")
                print(f"real prob: {real_prob}")

                # save the subset and info for the remained cross labels.
                remained_cross_labels[i + "-" + c] = {"data": rdf[
                    (rdf["dialogue_act"] == da_decode[c]) & (rdf[motive_decode[i[0].lower()] + " motive"] == True)],
                                              "prob": real_prob,
                                              "count": round(sum_df.at[i, c], 3)}

                # Only conduct domain adaption topic modeling with intersected label that has prob higher than the threshold.
                if real_prob > 0.1:
                    topic_modeling_target.append(i + "-" + c)
                    print("add for topic modelling!")
    return remained_cross_labels, topic_modeling_target

# grid search for optimized topic modelling params
def optimize_topic_modelling(topic_modeling_target, remained_cats, speaker_strings):
    result_dic = {}
    for target in topic_modeling_target:
        print("*" * 70)
        print(f"target category: {target}")
        subdf = remained_cats[target]["data"]
        sentences = subdf["reason"].tolist()
        sentences = preprocess(sentences, speakers_strings=speaker_strings)
        print_top_freq_words(sentences)
        print("Topic modelling params search!")
        best_params = find_best_topics(sentences)
        topic_model, topics, coherence = model_topics(sentences, best_params['umap_n_neighbor'],
                                                      best_params['umap_min_dist'], best_params['n_clusters'])
        topic_model.visualize_barchart()
        show_topics_example(topic_model, topics, subdf["reason"].tolist())
        result_dic[target] = {"topic_model": topic_model.get_topic_info(), "topics": topics, "coherence": coherence}


def main(source_result_folder):

    # load the whow annotated results
    results = load_results(source_result_folder)

    # find the speaker string to reduce topic sensitivity to speaker's name.
    speaker_strings = get_speaker_strings(results)

    # concatenate all whow annotated df
    rdf = pd.concat([df for eps, df in results.items()])

    # convert the result into sequence data.
    inter_utterance_sequence, intra_utterance_sequence = aggregate_utterance_labels(results)

    # calculate the frequency sum of the co-occurence dialogue x motive typs (e.g. social probing)
    sum_df, eps_label_counts = get_motive_dialogue_act_matrix_episode_breakdown(intra_utterance_sequence, index=index,
                                                                                col=col, normalise=False)

    # calculate the episode frequency mean of the co-occurence dialogue x motive typs (e.g. social probing)
    mean_df, eps_label_counts = get_motive_dialogue_act_matrix_episode_breakdown(intra_utterance_sequence, index=index,
                                                                                 col=col, normalise=True)

    # identify cross labels (e.g. social motive + probing) that occur more frequently than threshold.
    remained_cats, topic_modeling_target = identify_prominent_action_motive_pairs(mean_df, sum_df, rdf)

    # grid search for optimized topic modelling params and ranges.
    optimize_topic_modelling(topic_modeling_target[-1:], remained_cats, speaker_strings)



if __name__ == "__main__":
    whow_result_folder = "../../data/whow_annotated/"
    main(whow_result_folder)
