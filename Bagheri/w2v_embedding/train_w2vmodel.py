from gensim.models import Word2Vec
import os.path
import pickle
import sys
import nltk

newmode = "sql"
# newmode = ["sql", "xsrf", "xss", "command_injection",, "open_redirect", "path_disclosure", "remote_code_execution"]
#nltk.download('punkt')
all_words = []
mode = "withString"
if (len(sys.argv) > 1):
    mode = sys.argv[1]

print("Loading " + mode)
with open('corpus/corpus' + '_' + mode + "_X_" + newmode, 'r', encoding="utf-8") as file:
    pythondata = file.read().lower().replace('\n', ' ')
if (os.path.isfile('processed_data/pythoncorpus_processed_' + mode + '_' + newmode)):
    with open('processed_data/pythoncorpus_processed_' + mode + '_' + newmode, 'rb') as fp:
        all_words = pickle.load(fp)
    print("loaded processed model.")
else:
    print("now processing...")
    processed = pythondata
    all_sentences = nltk.sent_tokenize(processed)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    print("saving")
    with open('processed_data/pythoncorpus_processed' + mode + '_' + newmode, 'wb') as fp:
        pickle.dump(all_words, fp)
print("processed.\n")

# 尝试不同的模型参数
for mincount in [5]:
    for mincount in [10, 30, 50, 100, 300, 500, 5000]:
        for iterationen in [1, 5, 10, 30, 50, 100]:
            for s in [5, 10, 15, 30, 50, 75, 100, 200, 300]:
                print("\n\n" + mode + " W2V model with min count " + str(mincount) + " and " + str(
                    iterationen) + " Iterationen and size " + str(s))
                fname = "../model/" + newmode + "/word2vec_" + mode + str(mincount) + "-" + str(
                    iterationen) + "-" + str(s) + ".model"
                if (os.path.isfile(fname)):
                    print("model already exists.")
                    continue
                else:
                    print("calculating model...")
                    # training the model
                    model = Word2Vec(all_words, vector_size=s, min_count=mincount, epochs=iterationen,
                                     workers=6)
                    vocabulary = model.wv
                    print(vocabulary)
                    # print some examples
                    words = ["if"]
                    for similar in words:
                        try:
                            print("\n")
                            print(similar)
                            sim_words = model.wv.most_similar(similar)
                            print(sim_words)
                            print("\n")
                        except Exception as e:
                            print(e)
                            print("\n")
                    model.save(fname)