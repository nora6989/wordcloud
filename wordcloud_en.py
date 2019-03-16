import nltk
nltk.download('book', quiet=True)
from nltk.book import *

nltk.corpus.gutenberg.fileids()
emma_raw = nltk.corpus.gutenberg.raw("austen-emma.txt")
print(emma_raw[:1302])

from nltk.tokenize import sent_tokenize
print(sent_tokenize(emma_raw[:1000])[3])

from nltk.tokenize import word_tokenize
word_tokenize(emma_raw[50:100])

from nltk.tokenize import RegexpTokenizer
retokenize = RegexpTokenizer("[\w]+")
retokenize.tokenize(emma_raw[50:100])

words = ['lives', 'crying', 'flies', 'dying']
from nltk.stem import PorterStemmer
st = PorterStemmer()
[st.stem(w) for w in words]

from nltk.stem import LancasterStemmer
st = LancasterStemmer()
[st.stem(w) for w in words]

from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
[lm.lemmatize(w) for w in words]

lm.lemmatize("dying", pos="v")

nltk.help.upenn_tagset('VB')
from nltk.tag import pos_tag
sentence = "Emma refused to permit us to obtain the refuse permit"
tagged_list = pos_tag(word_tokenize(sentence))
tagged_list

nouns_list = [t[0] for t in tagged_list if t[1] == "NN"]
nouns_list

from nltk.tag import untag
untag(tagged_list)

def tokenizer(doc):
    return ["/".join(p) for p in tagged_list]

tokenizer(sentence)

from nltk import Text

text = Text(retokenize.tokenize(emma_raw), name="Emma")

text.plot(20)
plt.show()

text.dispersion_plot(["Emma", "Knightley", "Frank", "Jane", "Harriet", "Robert"])

text.concordance('Emma', lines=5)

text.similar('Emma', 10)

text.collocations(10)

fd = text.vocab()
type(fd)

from nltk import FreqDist

stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
emma_tokens = pos_tag(retokenize.tokenize(emma_raw))
names_list = [t[0] for t in emma_tokens if t[1] == "NNP" and t[0] not in stopwords]
fd_names = FreqDist(names_list)

fd_names.N(), fd_names["Emma"], fd_names.freq("Emma")

fd_names.most_common(5)