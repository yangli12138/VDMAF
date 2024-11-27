from gensim.models import Word2Vec, KeyedVectors
import os.path
import pickle
import sys
import nltk


#nltk.download('punkt')
all_words = []


print("Loading ")
with open('corpus', 'r', encoding="utf-8") as file:
    Cdata = file.read().lower().replace('\n', ' ')

print("Length of the training file: " + str(len(Cdata)) + ".")
print("It contains " + str(Cdata.count(" ")) + " individual code tokens.")

if (os.path.isfile('Ccorpus_processed')):
    with open('Ccorpus_processed', 'rb') as fp:
        all_words = pickle.load(fp)
    print("loaded processed model.")
else:
    print("now processing...")
    processed = Cdata
    all_sentences = nltk.sent_tokenize(processed)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    print("saving")
    with open('Ccorpus_processed', 'wb') as fp:
        pickle.dump(all_words, fp)

print("processed.\n")

# 尝试不同的模型参数
for mincount in [5, 30, 50, 100, 300, 500, 5000]:
    for iterationen in [100, 10]:
        for s in [300]:
            print("\n\n"  + " W2V model with min count " + str(mincount) + " and " + str(
                iterationen) + " Iterationen and size " + str(s))
            fname = "Reveal_word2vec_" + str(mincount) + "-" + str(iterationen) + "-" + str(s) + ".model"

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