# 🔄 Recurrent Neural Networks (RNN)

> The first major breakthrough in sequence modeling that paved the way for LSTMs, Transformers, and modern Large Language Models.

![Topic](https://img.shields.io/badge/Topic-RNN-blue)
![Domain](https://img.shields.io/badge/Domain-NLP-orange)
![Level](https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green)

---

## 🚀 Learning Path

This repository follows the evolution of Natural Language Processing architectures:

```text
NLP Basics
    ↓
Bag of Words
    ↓
RNN
    ↓
LSTM
    ↓
GRU
    ↓
Attention
    ↓
Transformers
    ↓
Large Language Models
```

---

## 📌 What is RNN?

A **Recurrent Neural Network (RNN)** is a deep learning architecture designed to process sequential data by maintaining a memory of previous inputs through hidden states.

Unlike traditional neural networks, RNNs can learn temporal relationships and contextual dependencies within a sequence.

### Key Characteristics

✅ Handles Sequential Data

✅ Maintains Hidden Memory

✅ Captures Context

✅ Learns Temporal Dependencies

✅ Foundation for LSTM and GRU

---

## 🤔 Why Do We Need RNNs?

Traditional Artificial Neural Networks assume that all inputs are independent.

However, many real-world tasks depend on previous information.

### Example

Sentence 1:

```text
I love this movie
```

Sentence 2:

```text
This movie love I
```

Both sentences contain the same words, but the order changes the meaning.

Traditional models struggle with this problem because they ignore sequence information.

RNNs solve this by remembering previous inputs while processing the sequence.

---

## 📚 What is Sequential Data?

Sequential data is information where the order of observations matters.

### Common Examples

| Application                | Example              |
| -------------------------- | -------------------- |
| 💬 Chatbots                | Conversational AI    |
| 🌐 Machine Translation     | Google Translate     |
| 🎙️ Speech Recognition     | Siri, Alexa          |
| ✍️ Text Generation         | Story Generation     |
| 📈 Time Series Forecasting | Stock Prediction     |
| 😊 Sentiment Analysis      | Movie Reviews        |
| 🔤 Next Word Prediction    | Keyboard Suggestions |

---

## ⚠️ Limitation of Bag of Words

Before deep learning became popular, text was often represented using the **Bag of Words (BoW)** approach.

Example:

```text
The food is good
```

Vocabulary:

```text
food, good, bad, not
```

Vector Representation:

```text
[1, 1, 0, 0]
```

### Problems with Bag of Words

❌ Loses word order information

❌ Cannot understand context

❌ Cannot capture long-term dependencies

❌ Treats different sentences similarly

These limitations motivated the development of sequence-based architectures like RNNs.

---

## ⚔️ ANN vs RNN

| Feature               | ANN     | RNN    |
| --------------------- | ------- | ------ |
| Memory                | ❌       | ✅      |
| Sequence Awareness    | ❌       | ✅      |
| Context Understanding | ❌       | ✅      |
| NLP Applications      | Limited | Strong |
| Temporal Learning     | ❌       | ✅      |

---

## 🏗️ RNN Architecture

An RNN processes data one timestep at a time while maintaining a hidden state.

```text
x₁ → h₁ → y₁
      ↓
x₂ → h₂ → y₂
      ↓
x₃ → h₃ → y₃
      ↓
x₄ → h₄ → y₄
```

Where:

* xₜ = Input at timestep t
* hₜ = Hidden State
* yₜ = Output at timestep t

The hidden state acts as the memory of the network and carries information from previous timesteps.

---

## 🧠 Hidden State Memory

At every timestep:

1. Current input is received.
2. Previous hidden state is combined with the input.
3. New hidden state is generated.
4. Output is produced.
5. Hidden state is passed forward.

This mechanism enables RNNs to learn patterns across sequences.

---

## ⚙️ Forward Propagation

### Hidden State Update

```text
hₜ = tanh(Wxh·xₜ + Whh·hₜ₋₁ + b)
```

### Output Calculation

```text
yₜ = Why·hₜ + by
```

### Components

| Symbol | Meaning                  |
| ------ | ------------------------ |
| Wxh    | Input-to-Hidden Weights  |
| Whh    | Hidden-to-Hidden Weights |
| Why    | Hidden-to-Output Weights |
| b      | Bias                     |
| hₜ     | Hidden State             |
| yₜ     | Output                   |

The same weights are reused at every timestep, enabling parameter sharing.

---

## 🔄 Backpropagation Through Time (BPTT)

RNNs are trained using **Backpropagation Through Time (BPTT)**.

### Process

1. Unfold the network across timesteps.
2. Perform forward propagation.
3. Compute loss.
4. Propagate gradients backward through time.
5. Update weights using gradient descent.

```text
t₁ ← t₂ ← t₃ ← t₄ ← t₅
```

This allows the network to learn relationships across an entire sequence.

---

## 🚨 Vanishing Gradient Problem

One of the biggest limitations of Simple RNNs is the **Vanishing Gradient Problem**.

As sequence length increases:

```text
Input → Hidden → Hidden → Hidden → Hidden → ...
```

Gradients become extremely small during backpropagation.

### Consequences

❌ Long-term information is forgotten

❌ Earlier context loses influence

❌ Training becomes difficult

❌ Performance degrades on long sequences

---

## 💡 Solution: LSTM and GRU

To overcome the Vanishing Gradient Problem, more advanced recurrent architectures were developed.

### 🔹 LSTM (Long Short-Term Memory)

* Memory Cells
* Forget Gate
* Input Gate
* Output Gate
* Better Long-Term Memory

### 🔹 GRU (Gated Recurrent Unit)

* Simplified LSTM
* Fewer Parameters
* Faster Training
* Strong Performance

These architectures significantly improved sequence modeling.

---

## 🌍 Real-World Applications of RNN

| Domain     | Use Case            |
| ---------- | ------------------- |
| NLP        | Sentiment Analysis  |
| NLP        | Language Modeling   |
| NLP        | Text Generation     |
| NLP        | Machine Translation |
| Speech     | Speech Recognition  |
| Healthcare | Patient Monitoring  |
| Finance    | Stock Forecasting   |
| Search     | Query Prediction    |

---

## 🎬 Practical Implementation

### MovieMood – Sentiment Analysis

An end-to-end NLP application that predicts whether a movie review is **Positive** or **Negative**.

### Concepts Used

* NLP Preprocessing
* Tokenization
* Sequence Padding
* Word Embeddings
* LSTM Networks
* Binary Classification
* Streamlit Deployment

### Repository

🔗 https://github.com/jainkhushi22/MovieMood-Sentiment-Analysis

This project demonstrates how sequence models can be applied to solve real-world NLP problems.

---

## 📈 Evolution of NLP Architectures

```text
Bag of Words
      ↓
Recurrent Neural Networks (RNN)
      ↓
Long Short-Term Memory (LSTM)
      ↓
Gated Recurrent Units (GRU)
      ↓
Attention Mechanism
      ↓
Transformers
      ↓
BERT / GPT
      ↓
Large Language Models
```

Understanding RNNs provides the foundation for understanding modern Generative AI systems.

---

## 🔜 What's Next?

The next module explores:

### Long Short-Term Memory (LSTM)

Topics Covered:

* Forget Gate
* Input Gate
* Output Gate
* Cell State
* Long-Term Dependencies
* Text Generation
* Next Word Prediction

---

## 🎯 Key Takeaways

✅ Designed for Sequential Data

✅ Maintains Hidden Memory

✅ Learns Contextual Relationships

✅ Uses Backpropagation Through Time

❌ Suffers from Vanishing Gradient Problem

✅ Inspired LSTM and GRU

✅ Foundation of Modern NLP

---

## 📖 References

* Stanford CS224N
* Deep Learning Specialization
* Krish Naik NLP Playlist
* Research Papers on Recurrent Neural Networks

---

## 👩‍💻 Author

**Khushi Jain**

AI & Data Science Engineering Student passionate about NLP, Deep Learning, Generative AI, and Large Language Models.

⭐ If you find this repository helpful, consider giving it a star and following the journey from RNNs to Transformers.
