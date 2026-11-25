from typing import List
import random
import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet dictionary")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

_global_lemmatizer = WordNetLemmatizer()

class Text:
    def __init__(self, label: str, raw_text: str, lemmatize: bool = False):
        self.label = label
        
        if lemmatize:
            words = raw_text.lower().split()
            self.text = " ".join([_global_lemmatizer.lemmatize(w) for w in words])
        else:
            self.text = raw_text

    def __repr__(self):
        return f"[{self.label}] {self.text}"

def read(filename: str, lemmatize: bool = False) -> List[Text]:
    texts = []
    with open(filename, 'r', encoding='latin-1') as file:
        for line in file:
            clean_line = line.strip()
            if not clean_line:
                continue
                
            parts = clean_line.split(" ", 1)
            if len(parts) == 2:
                label_block = parts[0]
                question_text = parts[1]
                fine_label = label_block
                
                text_obj = Text(fine_label, question_text, lemmatize=lemmatize)
                texts.append(text_obj)
                
    return texts

if __name__ == "__main__":
    file_path = "data/train_1000.label" 
    questions_data = read(file_path)
    random_samples = random.sample(questions_data, 10)
    for i in random_samples:
        print (i)
