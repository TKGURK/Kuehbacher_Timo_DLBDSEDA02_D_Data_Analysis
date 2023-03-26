# Standardbibliothek um eine CSV einzulesen
import csv
# Bereinigung des Textes
import re
# Herausfiltern von Stoppwörtern
from nltk.corpus import stopwords
# Vektorisierung mit CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Zweite Variante zur Vektorisierung: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Extrahieren der Themen
from sklearn.decomposition import NMF, LatentDirichletAllocation
#  Sentiment Analyse
from textblob import TextBlob

# 1. Datensatz (CSV-Datei) öffnen und einlesen
with open('Comcast.csv', newline='') as csvfile:
    raw_text = csv.DictReader(csvfile)

    # Liste für die später bereinigten Wörter
    no_stopword_list = []

    # 2. Text zeilenweise bereinigen
    for row in raw_text:
        # Auswählen der Spalte der CSV-Datei
        text_list = row['Customer Complaint']
        # Sonderzeichen, Satzzeichen raus und Text in String Objekt umwandeln
        text_list = re.sub('[^a-zA-Z0-9 ]', '', str(text_list))
        # Alle Buchstaben in Kleinbuchstaben umwandeln
        text_list = text_list.lower()
        # Wörter tokenisieren
        text_list = text_list.split()
        # Stoppwörter entfernen: Liste erstellen mit englischen Stoppwörtern
        stop_words_list = set(stopwords.words('english'))
        # Wörter in Stopword-freie Liste aufnehmen
        for word in text_list:
            if word not in stop_words_list:
                no_stopword_list.append(word)

# 3. Wörter in numerische Vektoren umwandeln
# Initialisierung des CountVectorizer
vectorizer = CountVectorizer()

# Anwendung des CountVectorizer auf die Wörter
vectors = vectorizer.fit_transform(no_stopword_list)
# Ausgabe der numerischen Vektoren zum Vergleich
# print(vectors)

# zweite Variante TF-IDF
# Initialisierung des TF-IDF-Vectorizer
vectorizer = TfidfVectorizer()

# Anwendung des TfidfVectorizer auf den Text
tfidf = vectorizer.fit_transform(no_stopword_list)
# Ausgabe der numerischen Vektoren zum Vergleich
# print(tfidf)

# 4. Themen extrahieren mit NMF und LDA
# Initialisierung von NMF
nmf = NMF(n_components=2, init='random', random_state=0)

# Anwendung von NMF auf den TF-IDF-Vektor
nmf_topic = nmf.fit_transform(tfidf)

# Ausgabe der Themen, die von NMF gefunden wurden
print("Themen, die von NMF gefunden wurden:")
for topic_idx, topic in enumerate(nmf.components_):
    print("Thema %d:" % (topic_idx))
    print(" ".join([vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[:-10 - 1:-1]]))

# Initialisierung von LDA
lda = LatentDirichletAllocation(n_components=2, max_iter=5,
                                learning_method='online',
                                learning_offset=50., random_state=0)

# Anwendung von LDA auf den TF-IDF-Vektor
lda_topic = lda.fit_transform(tfidf)

# Ausgabe der Themen, die von LDA gefunden wurden
print("Themen, die von LDA gefunden wurden:")
for topic_idx, topic in enumerate(lda.components_):
    print("Thema %d:" % (topic_idx))
    print(" ".join([vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[:-10 - 1:-1]]))

# 5. Sentiment Analyse
# Initialisierung des TextBlob-Objekts
blob = TextBlob(str(no_stopword_list))
# Sentiment-Analyse durchführen
sentiment = blob.sentiment.polarity
# Ausgabe des Sentiments
print("Sentiment:", sentiment)
