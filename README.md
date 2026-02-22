# AI Tutor: Your Intelligent Learning Companion 🎓

AI Tutor is a sophisticated RAG (Retrieval-Augmented Generation) application designed to transform how you interact with educational content. By leveraging the power of **Llama 3 (via Groq)**, **FAISS**, and **Hugging Face Embeddings**, it provides a seamless interface for deep document understanding, automated quiz generation, and structured learning paths.

![Project Demo](Demo.webm)

## 🚀 Core Features

### 1. Smart Study Assistant
Navigate through complex documents with ease.
- **Deep Summarization**: Extract core concepts from PDF and DOCX files instantly.
- **Contextual Q&A**: Ask technical or conceptual questions and receive precise answers backed by your document's context.
- **Semantic Retrieval**: Uses FAISS for high-speed, relevant information retrieval.

### 2. Intelligent Quiz Generator
Turn your study materials into interactive practice sessions.
- **Automatic Question Synthesis**: Generates high-quality multiple-choice questions based on your specific uploads.
- **Instant Evaluation**: Receive immediate feedback, scoring, and correct answer keys to reinforce learning.

### 3. Learning Roadmaps
Define your growth trajectory in the tech world.
- **Dynamic Curriculum Design**: Generate comprehensive, topic-based roadmaps for any technical domain.
- **Basics to Expert**: Structured paths that guide you from foundational concepts to advanced specializations.

---

## 🛠️ Technical Stack & Comparison

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Orchestration** | LangChain / LangGraph | State-aware RAG flows & modularity |
| **LLM Inference** | Groq LPU (Llama 3) | Sub-second latency for interactive feedback |
| **Vector Index** | FAISS | High-performance in-memory similarity search |
| **Embeddings** | all-MiniLM-L6-v2 | Lightweight yet semantic document vectors |
| **UI Framework** | Streamlit | Rapid, Python-native web interface |

---

## 📋 Getting Started

### Prerequisites
- Python 3.10 or higher
- A Groq API Key (get it at [console.groq.com](https://console.groq.com/))

### Installation & Setup

1. **Clone the Project**
   ```bash
   git clone https://github.com/SaiBuvanesh/AI-Tutor_Retrieval-Augmented-Learning-System.git
   cd AI-Tutor_Retrieval-Augmented-Learning-System
   ```

2. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Launch the Application**
   For Windows users, simply run:
   ```bash
   run.bat
   ```
   Alternatively, via terminal:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

---

## 🏛️ Project Structure

```
AI-Tutor/
├── app.py                  # Homepage & Navigation
├── pages/                  # Modular Application Features
│   ├── 1_Study_Assistant.py
│   ├── 2_Quiz_Generator.py
│   └── 3_Roadmaps.py
├── utils/                  # Core Logic & Helpers
│   ├── files.py            # Text Processing Pipeline
│   └── llm.py              # Model Initializations
├── requirements.txt        # System Dependencies
└── run.bat                 # Windows Launcher
```

---

Sai Buvanesh
