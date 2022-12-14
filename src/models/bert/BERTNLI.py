'''
NLI = natural language inference, i.e., entailment, contradictory, etc

sourcecode: https://keras.io/examples/nlp/semantic_similarity_with_bert/
'''

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import torch
SEED = 42
np.random.seed(SEED)
import random, os
from exp import result_summariser_dmresults
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE=32
#Create a custom data generator
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=BATCH_SIZE,
            bert_model="bert-base-uncased",
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        batches = len(self.sentence_pairs) // self.batch_size
        if batches * self.batch_size < len(self.sentence_pairs):
            return batches + 1
        else:
            return batches

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        if end > len(self.sentence_pairs):
            end = len(self.sentence_pairs)
        indexes = self.indexes[start: end]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


# There are more than 550k samples in total; we will use 100k for this example.
def read_data(in_dir):
    # train_df = pd.read_csv(in_dir+"/snli_1.0_train.csv", nrows=NROWS)
    # valid_df = pd.read_csv(in_dir+"/snli_1.0_dev.csv")
    # test_df = pd.read_csv(in_dir+"/snli_1.0_test.csv")
    train_df = pd.read_csv(in_dir + "/small.csv")
    valid_df = pd.read_csv(in_dir + "/small.csv")
    test_df = pd.read_csv(in_dir + "/small.csv")
    return train_df, valid_df, test_df




def one_hot_encoding(train_df, valid_df, test_df, label_match:str, label_nomatch:str):
    # One-hot encode training, validation, and test labels.
    train_df["label"] = train_df["similarity"].apply(
        lambda x: 0 if x == label_nomatch else 1 if x == label_match else 2
    )
    y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

    valid_df["label"] = valid_df["similarity"].apply(
        lambda x: 0 if x == label_nomatch else 1 if x == label_match else 2
    )
    y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

    test_df["label"] = test_df["similarity"].apply(
        lambda x: 0 if x == label_nomatch else 1 if x == label_match else 2
    )
    y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)

    return y_train, y_val, y_test


if __name__ == "__main__":
    max_length = 128 #int(sys.argv[4]) # Maximum length of input sentence to the model.
    batch_size = 32 #int(sys.argv[5])
    epochs = 1 #int(sys.argv[6])

    # Labels in our dataset.
    label_match="entailment"
    label_nomatch="contradiction"
    in_dir = sys.argv[1]
    bert_model=sys.argv[2]
    out_dir=sys.argv[3]

    setting = in_dir
    if "/" in in_dir:
        setting = in_dir[setting.rindex("/") + 1:]
    setting = setting+"_"+bert_model
    if "/" in bert_model:
        setting=setting[setting.rindex("/")+1:]
    setting=setting+"_"+str(max_length)+"_"+str(batch_size)+"_"+str(epochs)

    #read the datasets
    train_df, valid_df, test_df=read_data(in_dir)

    #1 hot encoding datasets
    y_train, y_val, y_test=one_hot_encoding(train_df, valid_df, test_df, label_match, label_nomatch)

    # Shape of the data
    print(f"Total train samples : {train_df.shape[0]}")
    print(f"Total validation samples: {valid_df.shape[0]}")
    print(f"Total test samples: {valid_df.shape[0]}")

    print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
    print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
    print(f"Similarity: {train_df.loc[1, 'similarity']}")

    # We have some NaN entries in our train data, we will simply drop them.
    print("Number of missing values")
    print(train_df.isnull().sum())
    train_df.dropna(axis=0, inplace=True)

    print("Train Target Distribution")
    print(train_df.similarity.value_counts())

    print("Validation Target Distribution")
    print(valid_df.similarity.value_counts())

    #The value "-" appears as part of our training and validation targets. We will skip these samples.
    #todo: this can be deleted
    train_df = (
        train_df[train_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    valid_df = (
        valid_df[valid_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


    #Build the model
    # Create the model under a distribution strategy scope.
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Encoded token ids from BERT tokenizer.
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model.
        bert_model = transformers.TFBertModel.from_pretrained(bert_model)
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        sequence_output, pooled_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        dropout = tf.keras.layers.Dropout(0.3)(concat)
        output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )


    print(f"Strategy: {strategy}")
    model.summary()

    #Create train and validation data generators
    train_data = BertSemanticDataGenerator(
        train_df[["sentence1", "sentence2"]].values.astype("str"),
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data = BertSemanticDataGenerator(
        valid_df[["sentence1", "sentence2"]].values.astype("str"),
        y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    #Train the Model
    #Training is done only for the top layers to perform "feature extraction", which will allow the model to use the
    #representations of the pretrained model.
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
    )

    #Fine-tuning
    #This step must only be performed after the feature extraction model has been trained to convergence on the new data.
    #This is an optional last step where bert_model is unfreezed and retrained with a very low learning rate. This can
    #deliver meaningful improvement by incrementally adapting the pretrained features to the new data.
    #Unfreeze the bert_model.
    bert_model.trainable = True
    #Recompile the model to make the change effective.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy", #todo: if task is binary, this needs changing
        metrics=["accuracy"],
    )
    model.summary()


    #Train the entire model end-to-end
    # history = model.fit(
    #     train_data,
    #     validation_data=valid_data,
    #     epochs=epochs,
    #     use_multiprocessing=True,
    #     workers=-1,
    # )

    #Evaluate model on the test set
    test_data = BertSemanticDataGenerator(
        test_df[["sentence1", "sentence2"]].values.astype("str"),
        y_test,
        batch_size=batch_size,
        shuffle=False,
    )

    print("<<<<< NOW EVALUATE >>>>>")
    #model.evaluate(test_data, verbose=0)
    pred = model.predict(test_data)
    pred=pred.argmax(axis=-1)
    p, r, f1=result_summariser_dmresults.save_scores(pred, y_test.argmax(1),
                                                     setting, 3, out_dir)

    print(pred)
    print("end")

# #Inference on custom sentences
# def check_similarity(sentence1, sentence2):
#     sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
#     test_data = BertSemanticDataGenerator(
#         sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
#     )
#
#     proba = model.predict(test_data)[0]
#     idx = np.argmax(proba)
#     proba = f"{proba[idx]: .2f}%"
#     pred = labels[idx]
#     return pred, proba
#
# #Check results on some example sentence pairs.
# sentence1 = "Two women are observing something together."
# sentence2 = "Two women are standing with their eyes closed."
# check_similarity(sentence1, sentence2)
#
# sentence1 = "A smiling costumed woman is holding an umbrella"
# sentence2 = "A happy woman in a fairy costume holds an umbrella"
# check_similarity(sentence1, sentence2)
#
# sentence1 = "A soccer game with multiple males playing"
# sentence2 = "Some men are playing a sport"
# check_similarity(sentence1, sentence2)




