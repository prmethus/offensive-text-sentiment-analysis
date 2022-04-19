# Downloading the datasets
def download_and_join_datasets():
    
    import os, requests, zipfile, glob
    import pandas as pd
    
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
    if not os.path.exists(os.path.join(datasets_dir, "Cyberbully_dataset_combined.csv")):
        print("Joining the datasets together...")
        datasets = glob.glob(os.path.join(datasets_dir, "*.csv"))
        dataframe = pd.concat(list(map(lambda dataset: pd.read_csv(dataset)[["Text", "oh_label"]], datasets)),axis=0)    
        dataframe = dataframe.rename(columns={"oh_label":"Offensive"})
        print("Dataset Joined Successfully. Saving it...")
        dataframe.to_csv(os.path.join(datasets_dir, "Cyberbully_dataset_combined.csv"), index=False)
        print(r"Saved 'Cyberbully_dataset_combined.csv'. The dataset is also availabe on Kaggle: https://www.kaggle.com/datasets/prmethus/mendeleys-cyberbully-datasets-combined?select=Cyberbully_dataset_combined.csv")
        
    print(f"\nDataset path: {os.path.join(datasets_dir, 'Cyberbully_dataset_combined.csv')}")

            
# Defining a function for tokenizin and padding words in a sentence
def tokenize_and_pad(texts, tokenizer, maxlen=100, padding='post', truncating='post'):
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen, padding=padding, truncating=truncating)
    
    return padded_sequences


def stratified_split(dataset):
        
    from sklearn.model_selection import StratifiedShuffleSplit
    
    dataset = dataset.reset_index() # Required for Stratified Split.
    
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    
    for train_set_index, validation_test_set_index in split1.split(dataset["Text"], dataset["Offensive"]):
        train_set = dataset.loc[train_set_index,["Text", "Offensive"]]
        validation_test_set = dataset.loc[validation_test_set_index,["Text", "Offensive"]]
    
    validation_test_set = validation_test_set.reset_index()
    
    for validation_set_index, test_set_index in split2.split(validation_test_set["Text"], validation_test_set["Offensive"]):
        validation_set = validation_test_set.loc[validation_set_index,["Text", "Offensive"]]
        test_set = validation_test_set.loc[test_set_index,["Text", "Offensive"]]
        
    return train_set, validation_set, test_set