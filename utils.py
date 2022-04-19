# Downloading the datasets
def download_dataset():
    import os, requests, zipfile
    datasets_zip = "cyberbullying_dataset-mendeley_data.zip"
    datasets_dir = "cyberbully_datasets/"
    datasets_url = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/jf4pzyvnpj-1.zip"

    if not os.path.exists("cyberbullying_dataset-mendeley_data.zip"):
        print("The dataset doesn't exist. Downloading it...")
        with open("cyberbullying_dataset-mendeley_data.zip", "wb") as rf:
            rf.write(requests.get(datasets_url).content)
            print("Dataset successfully downloaded")
    else:
        print("The Dataset already exists!")

    if not os.path.exists(datasets_dir):
        print("Extracting the zip file...")
        os.mkdir(datasets_dir)
        with zipfile.ZipFile(datasets_zip) as zf:
            zf.extractall(datasets_dir)
            print("The Dataset has been extracted!")

            
# Defining a function for tokenizin and padding words in a sentence
def tokenize_and_pad(texts, tokenizer, maxlen=100, padding='post', truncating='post'):
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen, padding=padding, truncating=truncating)
    
    return padded_sequences

def stratified_sampling_train_test_validation(sentences, labels):
    
    from sklearn.model_selection import StratifiedShuffleSplit
    
    s1 = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    s2 = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
    
    for train_index, test_valid_index in s1.split(sentences, labels):
        sentences_for_training, labels_for_training = sentences[train_index], labels[train_index]
        sentences_for_testing_valid, labels_for_testing_valid = sentences[test_valid_index], labels[test_valid_index]
        
    for validation_index, test_index in s2.split(sentences_for_testing_valid, labels_for_testing_valid):
        sentences_for_validation, labels_for_validation = sentences_for_testing_valid[validation_index], labels[validation_index]
        sentences_for_testing, labels_for_testing = sentences_for_testing_valid[test_index], labels[test_index]
        
        return (sentences_for_training, labels_for_training), (sentences_for_validation, labels_for_validation), (sentences_for_testing, labels_for_testing)