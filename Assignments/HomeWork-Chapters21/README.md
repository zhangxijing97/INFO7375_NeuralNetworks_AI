# HW to Chapter 21 “NLP and Machine Translation”

# Non-programming Assignment

## 1. Describe word embedding.
Word embedding is a technique in natural language processing (NLP) where words are represented as dense vectors in a continuous vector space. These vectors capture semantic and syntactic relationships between words by placing similar words closer to each other in the vector space. Examples of word embedding models include Word2Vec, GloVe, and FastText. Word embeddings help reduce the dimensionality of text data while preserving meaningful relationships between words.

---

## 2. What is the measure of word similarity?
Word similarity is measured by the closeness of word vectors in the embedding space. Common measures include:
- **Cosine Similarity**: Computes the cosine of the angle between two word vectors. A value closer to 1 indicates higher similarity.
- **Euclidean Distance**: Measures the straight-line distance between two vectors. Smaller distances indicate higher similarity.
- **Dot Product**: Measures similarity based on the product of two vectors, often used in some models.

---

## 3. Describe the Neural Language Model.
A Neural Language Model (NLM) uses neural networks to predict the probability distribution of word sequences. It typically includes:
1. **Input Layer**: Takes word representations (e.g., embeddings) as input.
2. **Hidden Layers**: Processes the input using architectures like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), or Transformer-based models.
3. **Output Layer**: Predicts the next word or sequence of words using a softmax function over the vocabulary.
Modern NLMs, like GPT and BERT, leverage self-attention mechanisms to capture long-range dependencies and context.

---

## 4. What is a bias in word embedding, and how to do debiasing?
**Bias in Word Embedding**: Word embeddings can capture and amplify societal biases present in the training data, such as gender or racial stereotypes. For example, embeddings may associate "man" with "engineer" and "woman" with "nurse" due to biased datasets.

**Debiasing Methods**:
1. **Hard Debiasing**: Projects biased dimensions onto a neutral subspace and removes the bias component.
2. **Soft Debiasing**: Adjusts word vector representations to reduce bias while preserving useful properties.
3. **Bias Regularization**: Incorporates bias correction into the training process to mitigate bias during embedding generation.

---

## 5. How does modern machine translation work using the language model?
Modern machine translation leverages neural networks, particularly Transformer-based architectures, such as the encoder-decoder model. The process involves:
1. **Encoding**: The source text is encoded into a dense representation using the encoder.
2. **Attention Mechanism**: Self-attention layers capture relationships between words in the context of the entire sentence.
3. **Decoding**: The decoder generates the target language sequence word by word, guided by the encoder's output.
Popular models like Google’s Neural Machine Translation (GNMT) and OpenAI’s GPT utilize this framework.

---

## 6. What is beam search?
Beam search is a heuristic search algorithm used in sequence generation tasks, such as machine translation. It works by:
1. Maintaining a fixed number of candidate sequences (beam width) at each step.
2. Expanding each sequence by appending possible next words.
3. Keeping only the top-scoring sequences based on a scoring function.
Beam search helps balance exploration and exploitation, leading to higher-quality translations compared to greedy decoding.

---

## 7. What is the BLEU score?
The **BLEU (Bilingual Evaluation Understudy)** score is a metric for evaluating the quality of machine-translated text compared to human reference translations. It measures:
- **N-gram Precision**: The overlap of n-grams between the machine translation and reference.
- **Brevity Penalty**: Adjusts for overly short translations.
Scores range from 0 to 1, with higher scores indicating better translation quality. A BLEU score close to 1 suggests the translation closely matches the reference text.