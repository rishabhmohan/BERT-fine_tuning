# BERT: Bidirectional Encoder Representations from Transformers

### Understanding Transformers: 

https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html

https://jalammar.github.io/illustrated-transformer/

The Transformer was proposed in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
The main idea of Transformer was to combine the advantages from both CNNs and RNNs in a novel architecture using the attention mechanism. 
Transformer architecture achieves parallelization by capturing recurrence sequence with attention and at the same time encodes each item’s 
position in the sequence. As a result, it leads to a compatible model with significantly shorter training time.

### Word2vec
The purpose and usefulness of Word2vec is to group the vectors of similar words together in vectorspace.
That is, it detects similarities mathematically. Word2vec creates vectors that are distributed numerical representations of word features,
features such as the context of individual words. It does so in one of two ways, either using context to predict a target word 
(a method known as continuous bag of words, or CBOW), or using a word to predict a target context, which is called skip-gram.
We use the latter method because it produces more accurate results on large datasets. 
Unfortunately, this approach to word representation does not addres polysemy, or the co-existence of many possible meanings for a given word or phrase. 
One thing that ElMO and BERT demonstrate is that by encoding the context of a given word, by including information about preceding and succeeding words
in the vector that represents a given instance of a word, we can obtain much better results in natural language processing tasks.
BERT owes its performance to the attention mechanism.

### Why BERT better
* Аttention only model without RNNs (LSTM/GRU) is computationally more attractive (parallel rather than sequential processing of input) and even has better performance (ability remember information beyond just about 100+ words) than RNNs.
* BERT uses an idea of representing words as subwords or ngrams
* Eliminates the need for task specific architectures (with fine tuning)
* BERT can generate different word embeddings for a word that captures the context of a word — that is its position in a sentence (unlike Word2vec and Glove)
* BERT encodes context bidirectionally, while due to the autoregressive nature of language models, GPT only looks forward (left-to-right).
* They can be used for downstream tasks which have very little labeled data

### BERT steps (Use Huggingface’s transformers library) 
This library lets you import a wide range of transformer-based pre-trained models.
1. Split text to train and test , labels and data both
2. Import Bert model and Bert tokenizer (from transformers import AutoModel, BertTokenizerFast). Output of using tokenizer  is a dictionary of two items
    1. ‘input_ids’ contains the integer sequences of the input sentences. The integers 101 and 102 are special tokens. We add them to both the sequences, and 0 represents the padding token.
    2. ‘attention_mask’ contains 1’s and 0’s. It tells the model to pay attention to the tokens corresponding to the mask value of 1 and ignore the rest.
3. Tokenize with padding : Set the padding length as average text length ( BertTokenizerFast.from_pretrained('bert-base-uncased’).batch_encode_plus(..) is used for tokenization+ padding.
4. Tensor: Convert integer sequences (both input_ids and attention masks) to tensors. 
5. Create dataloaders for both train and validation set. These dataloaders will pass batches of train data and validation data as input to the model during the training phase.  
6. Freeze all the layers of the pertained BERT model before fine-tuning it. This will prevent updating of model weights during fine-tuning
7. Define BERT architecture with dropout layer, Relu layer, fc1, fc2, softmax and ADAM optimizer
8. Compute class weights for the labels in the train set and then pass these weights to the loss function so that it takes care of the class imbalance.
9. Fine tune the model meaning train the last layer . Classification tasks such as sentiment analysis are done similarly to Next Sentence classification, by adding a classification layer on top of the Transformer output for the [CLS] token

### Hyperparameters
1. The max_seq_length specifies the maximum number of tokens of the input. Remember seq_length is max 512 (no of tokens)
2. batch size The train batch size is a number of samples processed before the model is updated. Our motive is to utilize our resource fully. So, you should set train batch size to maximum value based on the available ram. Could be 32 or 512 etc
3. masked_lm_prob is the percentage of words that are replaced with a [MASK] token in each sequence. max_predictions_per_seq= (max_seq_length* masked_lm_prob)

### Word Masking (as BERT training includes no labeled data so we have to mask 15% of words)
Training the language model in BERT is done by predicting 15% of the tokens in the input, that were randomly picked. These tokens are pre-processed as follows — 80% are replaced with a “[MASK]” token, 10% with a random word, and 10% use the original word. The intuition that led the authors to pick this approach is as follows (Thanks to Jacob Devlin from Google for the insight):
* 		If we used [MASK] 100% of the time the model wouldn’t necessarily produce good token representations for non-masked words. The non-masked tokens were still used for context, but the model was optimized for predicting masked words.
* 		If we used [MASK] 90% of the time and random words 10% of the time, this would teach the model that the observed word is never correct.
* 		If we used [MASK] 90% of the time and kept the same word 10% of the time, then the model could just trivially copy the non-contextual embedding.


### The BERT model expects three inputs:
* 		The input ids — for classification problem, two inputs sentences should be tokenized and concatenated together (please remember about special tokens mentioned above)
* 		The input masks — allows the model to cleanly differentiate between the content and the padding. The mask has the same shape as the input ids, and contains 1 anywhere the the input ids is not padding.
* 		The input types (labels) — also has the same shape as the input ids, but inside the non-padded region, it contains 0 or 1 indicating which sentence the token is a part of.

### BERT: From Decoders to Encoders
- The openAI transformer gave us a fine-tunable pre-trained model based on the Transformer. But something went missing in this transition from LSTMs to Transformers. ELMo’s language model was bi-directional, but the openAI transformer only trains a forward language model. Could we build a transformer-based model whose language model looks both forward and backwards (in the technical jargon – “is conditioned on both left and right context”)? Yes, that’s BERT
- Finding the right task to train a Transformer stack of encoders is a complex hurdle that BERT resolves by adopting a “masked language model” concept from earlier literature (where it’s called a Cloze task).
- Beyond masking 15% of the input, BERT also mixes things a bit in order to improve how the model later fine-tunes. Sometimes it randomly replaces a word with another word and asks the model to predict the correct word in that position.


### BERT
1. The first step is to use the BERT tokenizer to first split the word into tokens. 
2. Add the special tokens needed for sentence classifications (these are [CLS] at the first position, and [SEP] at the end of the sentence).
3. Tokenizer replaces each token with its id from the embedding table which is a component we get with the trained model.
Note that the tokenizer does all these steps in a single line of code:
tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

4. Padding and processsing with BERT:
        ```input_ids = torch.tensor(np.array(padded))
        last_hidden_states = model(input_ids)```

5. After running this step, last_hidden_states holds the outputs of DistilBERT. It is a tuple with the shape (number of examples, max number of tokens in      the sequence, number of hidden units in the DistilBERT model). In our case, this will be N (no of data points), 66 (which is the number of tokens in        the longest sequence from the 2000 examples), 768 (the number of hidden units in the DistilBERT model).
6. For sentence classification, we’re only only interested in BERT’s output for the [CLS] token, so we select that slice of the cube and discard everything else. Slice the output for the first position for all the sequences, take all hidden unit outputs

```features = last_hidden_states[0][:,0,:].numpy()```

And now features is a 2d numpy array (no of rows * no of hidden units) containing the sentence embeddings of all the sentences in our dataset.
6. Split data: train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
7. Next, we train the Logistic Regression model on the training set.

```lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels```
