# Royal Enfield Sales & Test Drive AI 🏍️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_AI-orange)
![Gemini](https://img.shields.io/badge/AI-Gemini_2.0_Flash-green)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-purple)

**The Royal Enfield Sales AI** is a stateful conversational agent designed to secure test drive bookings for the all-new **Bullet 650**. Unlike simple chatbots, it uses a directed graph architecture to actively guide users from casual browsing to becoming a qualified lead, blending technical product knowledge with the emotional appeal of motorcycling heritage.

---

## Features

* **Intelligent Intent Detection**: Dynamically classifies user input into categories (Greeting, Technical Inquiry, Booking Intent) to route the conversation effectively.
* **Context-Aware RAG**: Answers specific questions (engine specs, refund policies, shipping) by retrieving data from a structured local knowledge base (txt files with header injection).
* **Smart Lead Capture**: Identifies high-intent riders and collects their **Name**, **Phone Number**, and **Location** for dealership assignment.
* **Contextual Memory**: Remembers conversation history across turns using LangGraph's memory, ensuring the bot knows the user's name or preferred city even after answering a technical question.
* **Modular Architecture**: Built with a clean `src/` structure, separating logic for vector storage, graph nodes, and configuration.

---

## Project Structure

```text
re-sales-bot/
│
├── data/                        # RAG Knowledge Base
│   ├── product_details.txt      # Bike specs (Engine, Chassis, Features)
│   ├── cancellation_policy.txt  # Test drive cancellation rules
│   ├── refund_policy.txt        # Booking refund details
│   └── shipping_policy.txt      # Delivery & logistics info
│
├── src/                         # Application Source Code
│   ├── __init__.py
│   ├── config.py                # Environment variables & paths
│   ├── vector_store.py          # ChromaDB initialization & embedding logic
│   ├── data_loader.py           # Custom chunking (Header Injection Strategy)
│   ├── retriever.py             # RAG retrieval logic
│   ├── nodes.py                 # Core agent functions (Sales Pitch, Extraction)
│   ├── state.py                 # StateGraph definitions & routing
│   └── graph.py                 # Graph compilation
│
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── .env                         # API Keys (Not included in repo)
└── README.md                    # Project documentation
```

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- A Google Cloud API Key (for Gemini)

### 2. Clone the Repository

```bash
git clone https://github.com/fahadNoufal/sales-agent-RE-650.git
cd sales-agent-RE-650
```

### 3. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a file named .env in the root directory and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the main script to start the interactive CLI bot:

```bash
python main.py
```

Note: The first time you run this, it will automatically:

1. Initialize the ChromaDB vector store.
2. Ingest the data from data/knowledgeBase.json.
3. Generate embeddings locally (this may take a few seconds).

Example Interaction:

```
You: hello
Bot: Hello! Ready to feel the evolution of a legend? The all-new Bullet 650 is finally here—combining the iconic silhouette you love with our award-winning 650cc twin engine. You really need to feel the torque to believe it. Can I get you booked for a test drive this week?

You: wait, how much power does it have?
Bot: The Bullet 650 pushes out 47.04 PS @ 7250 rpm. It’s a massive step up from the 350, giving you effortless highway cruising while keeping that signature thump. Ready to feel that power on the road?

You: what if i book a ride and can't make it?
Bot: No worries at all! According to our policy, you can cancel or reschedule your test drive at any time for free. Just keep in mind that if you're more than 30 minutes late, we might release the slot. Would you like me to book a flexible slot for you now?

You: sure, let's do it
Bot: Awesome! To lock in your ride, I just need your Full Name, Contact Number, and the City you are located in.

You: I'm Arjun from Bangalore
Bot: Thanks Arjun! Could you please share your contact number to finalize the booking?

You: 9876543210
Bot: Booking confirmed! Get ready to experience the legend.

[System Log]: Lead captured: Arjun, 9876543210, Bangalore
```

## Architecture Explained

This project uses **LangGraph** to manage the conversation flow as a flexible state machine, avoiding rigid loops.

![LangGraph Flow](resource/graph.png)

#### Design Philosophy:
Non-Blocking Flows Instead of trapping users in a rigid data-collection loop, this architecture re-evaluates user intent at every turn. This allows for non-linear conversations: users can pause the sign-up process to ask clarifying questions ("Is it free?") and resume seamlessly. This mimics human interaction, reduces frustration, and increases the likelihood of lead conversion by addressing doubts in real-time.

1.  **START**: User input is received.
2.  **Classify Intent**: At **every turn**, the `classify_user_intent` node analyzes the text and dynamically routes to one of four paths:
    * **Greeting**: Handles casual hellos and opens the conversation.
    * **Inquiry**: Uses **RAG** (ChromaDB + Gemini) to answer specific questions based on the JSON knowledge base.
    * **Lead**: Identifies high-intent users expressing interest in signing up.
    * **Extract Details**: parses user input to capture specific data slots (Name, Email, Platform). [design philosophy explained below]
3.  **Memory**: The graph utilizes `MemorySaver` to persist the conversation state, ensuring the bot remembers context (like a name mentioned earlier) even if the topic changes.
4. **Trigger Actions**: User lead information is recieved, it triggers a lead_captured function call.

## Customization

- Change the Knowledge Base: Edit policy_document folder to update product or policies. Delete the vector-db folder to force a rebuild on the next run.

- Switch LLM: Open src/config.py to change the model (e.g., from gemini-2.0-flash to gpt-4o via LangChain).

- Adjust Prompts: All system prompts are located in src/nodes.py.

## Prompt Engineering & Iteration

Current Prompt Strategy:
Intent Classifier: Uses a few-shot approach to categorize inputs without conversational filler.

RAG Product Specialist: Implements a "Ride-First" philosophy, converting dry specs into rider benefits while strictly forbidding hallucinations.

Lead Extraction Engine: Enforces a rigid JSON output format with null value handling for missing data.

## Evaluation Results
I evaluated the system using a custom 8-question rubric covering three critical domains.

Scoring Rubric:
- Accuracy: 95% (Grounding works perfectly with header-enriched metadata).
- Hallucination Avoidance: High (blocks irrelevant retrievals).
- Tone Consistency: High (Maintains "Modern Classic" persona).

##### What I’m most proud of: 

I am most proud of the Transition from RAG to Agent. While the assignment focused on retrieval, I implemented a Stateful LangGraph Agent that uses RAG not just to answer questions, but to actively drive a business goal: Lead Generation. The implementation of Header-Enriched Chunking ensures that the RAG context is never lost, even in complex multi-turn conversations. Also implemented Custom Logging, Full visibility into intent classification and RAG retrieval.

##### One thing I’d improve next: 
With more time, I would implement a better Evaluation for the system. Instead of manual testing, I’d try using an automated framework to score the 'Faithfulness' and 'Answer Relevancy' of the bot's responses against the policy documents to ensure 100% accuracy at scale. Also i would spend some time with the reranking of the retrieved documents.