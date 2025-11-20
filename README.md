# Agentic Mental Health Assistant

An intelligent, multi-agent mental health support system powered by **LangGraph** and **LangChain**. This assistant features emotional reasoning, specialized tools for academic anxiety, and robust safety guardrails for crisis intervention.

## 🎥 Demo Video

Watch the system in action here:  
**[View Demo Walkthrough](https://drive.google.com/file/d/1qjaVxKYWpPSm7meYxIBAZrjjFoVvPs3L/view?usp=drive_link)**

---

## 🚀 Key Features

* **🤖 Multi-Tool Orchestration**: Uses **LangGraph** to manage agent state, route queries, and handle interruptions seamlessly.
* **🧠 Emotional Reasoning**: Integrates **MedGemma** to provide medically grounded, empathetic responses for general mental health support.
* **📚 Exam Stress RAG Agent**: A specialized retrieval-augmented generation pipeline using **FAISS** and a custom dataset focused on academic anxiety and exam stress.
* **🚨 Safety & Crisis Intervention**:
    * **Risk Classification**: Automatically detects suicidal intent or high-risk phrases.
    * **Emergency Protocol**: Triggers an immediate emergency call via **Twilio** if high risk is detected.
    * **Fallback Prompts**: Ensures safe responses even when the LLM encounters uncertainty.
* **📍 Therapist Locator**: A utility tool to help users find professional help based on their location.

## 🛠️ Tech Stack

* **Orchestration**: LangChain, LangGraph
* **LLM**: MedGemma (via Ollama)
* **Vector Database**: FAISS
* **External APIs**: Twilio (Voice/SMS), Google Maps/SerpAPI (Locator)
* **Language**: Python

## ⚙️ Architecture

The system operates on a graph-based architecture:
1.  **Input Analysis**: User query is analyzed for intent and risk level.
2.  **Risk Guardrails**: If high risk (e.g., self-harm) is detected, the system bypasses standard flows and triggers the **Emergency Tool**.
3.  **Routing**: Safe queries are routed to the specific expert:
    * *Academic Stress* -> **Exam RAG Tool**
    * *Therapy Request* -> **Locator Tool**
    * *General Support* -> **MedGemma Agent**
4.  **Response**: The selected agent generates and delivers the response to the user.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/agentic-mental-health.git](https://github.com/yourusername/agentic-mental-health.git)
    cd agentic-mental-health
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your keys:
    ```env
    TWILIO_ACCOUNT_SID=your_sid
    TWILIO_AUTH_TOKEN=your_token
    TWILIO_FROM_NUMBER=your_twilio_number
    TWILIO_TO_NUMBER=emergency_contact_number
    OPENAI_API_KEY=your_key  # If using OpenAI for embeddings/routing
    SERPAPI_API_KEY=your_key # For therapist locator
    ```

4.  **Run the Application**
    ```bash
    python main.py
    # Or if using Streamlit/FastAPI
    streamlit run app.py
    ```

## ⚠️ Disclaimer

This project is an AI assistant for educational and supportive purposes only. It is **not** a replacement for professional medical advice, diagnosis, or treatment. In case of a real emergency, please contact local emergency services immediately.

---

*Built with ❤️ using LangChain & LangGraph*