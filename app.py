import torch
import torch.nn.functional as F
import json
import os

from flask import Flask
from flask import request
from flask import make_response
import pickle
from difflib import get_close_matches
app = Flask(__name__)

label_indexer = None
vocab_indexer = None
with open('label_indexer.pkl', 'rb') as i:
    label_indexer = pickle.load(i)
with open('vocab_indexer.pkl', 'rb') as i:
    vocab_indexer = pickle.load(i)
vocab = [s for s in vocab_indexer.objs_to_ints.keys()]
script = torch.jit.load('rnn_script.pt')

def get_closest_word(w):
    top_match = []
    diff_per = 0.100
    while not top_match:
        diff_per -= 0.05
        if diff_per == 0:
            # choose a random word if nothing matches! :(
            return int(random() * (len(vocab_indexer) - 1)) + 1
        top_match = get_close_matches(w, vocab, 1, diff_per)
    # print(f"closest match for '{w}' is '{top_match[0]}'")
    return vocab_indexer.index_of(top_match[0])

def clean_word(w):
    return ''.join([c for c in w if c.isalpha()])
    
@app.route('/codeswitch', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))
    words = req["text"]
    sent = [clean_word(s) for s in words.split()]
    word_idxs = []
    for w in sent:
        idx = vocab_indexer.index_of(w)
        if idx == -1:
            idx = get_closest_word(w)
        word_idxs.append(idx)
    test_input1 = torch.LongTensor([word_idxs])
    weights = script(test_input1)
    weights = weights.squeeze(0)
    probs = F.log_softmax(weights, dim=1)
    print("PROBS: ", probs)
    i = 0
    words = []
    labels = []
    for label in torch.argmax(probs[:, 1:], dim=1):
        words.append(sent[i])
        labels.append(label_indexer.get_object(label.item() + 1))
        i += 1
    ans = {
        "words": words,
        "labels": labels
    }
    r = make_response(json.dumps(ans))
    r.headers['Content-Type'] = 'application/json'
    return r


if __name__ == '__main__':
    app.run(debug=False, port=88, host='0.0.0.0', threaded=True)

