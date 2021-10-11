import logging
import json
import os
import shutil
from collections import defaultdict

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

def get_nlp():
    return camphr.load(
        """
    lang:
        name: ja_mecab
    pipeline:
        transformers_model:
            trf_name_or_path: bert-base-japanese
    """
    )

class EmbeddingBasedReplyAgent:
    def __init__(self, conversation_data_dir):
        self.conversation_data_dir = conversation_data_dir
        if os.path.exists(self.conversation_data_dir):
            self.load()

    def is_ready(self):
        return os.path.exists(self.conversation_data_dir)

    def load(self):
        print("Loading conversation data and model.")
        self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        self.u = AnnoyIndex(768, 'angular')
        self.u.load(os.path.join(self.conversation_data_dir, "index.ann"))
        self.conv_pairs = json.load(
            open(os.path.join(self.conversation_data_dir, "conversations.json")))
        print("Everything has been loaded.")

    def ingest_csv(self, path_to_conv_csv):
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        os.makedirs(self.conversation_data_dir, exist_ok=True)

        shutil.copy(path_to_conv_csv, os.path.join(
            self.conversation_data_dir, "conversation_pairs.csv"))
        
        conversation_examples = []
        for line in open(path_to_conv_csv, "r").readlines():
            conversation_examples.append(line.split(","))

        conversation_data = []

        for bob_talk, alice_reply in conversation_examples:
            embedding = model.encode(bob_talk)
            # import IPython; IPython.embed()
            conversation_data.append(dict(
                embedding=embedding.tolist(),
                bob_talk=bob_talk,
                alice_reply=alice_reply))

        json.dump(conversation_data, open(
            os.path.join(self.conversation_data_dir, "conversations.json"), 'w'))

        vec_dimension = len(conversation_data[0]['embedding'])
        t = AnnoyIndex(vec_dimension, 'angular')  # Length of item vector that will be indexed
        for i, conv_exam in enumerate(conversation_data):
            t.add_item(i, conv_exam['embedding'])

        print(f"Dimension is {vec_dimension}.")

        t.build(10) # 10 trees
        t.save(os.path.join(self.conversation_data_dir, 'index.ann'))

        self.load()

    def show_debug(self, mesage):
        st.write("## Top10 Reply Scores")
        top_10 = self.u.get_nns_by_vector(self.nlp(message).vector.tolist(), 10, include_distances=True)
        tmp_dict = defaultdict(list)
        for reply_id, score in zip(top_10[0], top_10[1]):
            tmp_dict["distance"].append(score)
            tmp_dict["Talk"].append(self.conv_pairs[reply_id]["bob_talk"])
            tmp_dict["reply"].append(self.conv_pairs[reply_id]["alice_reply"])
        st.table(pd.DataFrame(tmp_dict))

        st.write("## Conversation Pairs")
        st.table(pd.read_csv(
            os.path.join(
                self.conversation_data_dir, "conversation_pairs.csv")))
    def reply_to(self, message):
        closest, distance = self.u.get_nns_by_vector(self.model.encode(message).tolist(), 1, include_distances=True)
        # closest = top_10[0]
        # distance = top_10[1]
        # import IPython; IPython.embed()
        print(distance[0])
        reply = self.conv_pairs[closest[0]]
        return reply["alice_reply"]
