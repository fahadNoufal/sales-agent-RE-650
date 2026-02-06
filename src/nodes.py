import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.config import GEMINI_MODEL
from src.custom_logger import logging
from src.state import get_chat_history,State


llm = init_chat_model(GEMINI_MODEL)


def classify_user_enquiry_type(state:State) -> State:
    
    # retrieving past n messages by the user to find out the user intent
    # user_chat = state['messages']
    
    logging.info("Classifying user enquiry type based on chat history.")
    chat_history = str(get_chat_history(state))
    
    prompt = [ SystemMessage("""
                    ### ROLE
                    Expert Intent Classifier for a Bike Dealership.

                    ### CATEGORIES
                    - **greeting**: Social openers or pleasantries.
                    - **inquiry**: Questions about bike specs, warranty, or policies requiring document lookup.
                    - **lead**: High-intent actions such as booking a test drive or requesting a quote.
                    - **extract**: Providing contact info, location, or preferred bike model details.

                    ### CRITICAL OUTPUT RULE
                    - Output ONLY the category name.
                    - STRICTLY NO JSON, NO preamble, and NO conversational filler.

                    ### EXAMPLES
                    User: "Hi there!" -> greeting
                    User: "I want to book a test drive for the X-500 tomorrow." -> lead
                    User: "What is the top speed of the cruiser model?" -> inquiry
                    User: "My phone number is 555-0199." -> extract
                """),
              
              HumanMessage(chat_history)
            ]
    
    intent = llm.invoke(prompt)

    return {'user_intent': intent}

def reply_to_casual_greeting(state:State) -> State:
    
    # user_chat = state['messages'][-1].content
    
    logging.info("Generating reply to casual greeting.")
    chat_history = str(get_chat_history(state))
    
    
    prompt = [
        SystemMessage("""
                    **Role:** You are an expert Sales Agent for the all-new Royal Enfield Bullet 650. Your primary objective is to get the user to sign-up for a test drive.

                    **Persona:** Professional, passionate about motorcycling heritage, and helpful. Your tone should be "Modern Classic"—reverent of the Bullet’s 90-year legacy but excited about the new 650cc twin engine.

                    **Guidelines:**
                    1. **The Greeting:** When the user says "Hello" or greets you, respond warmly and immediately deliver a high-impact sales pitch for the Bullet 650. Focus on the legendary "thump" now paired with 650cc parallel-twin smoothness. End with a call to action to sign up for a test drive.
                    2. **The Value Hook:** provide brief, punchy "value bombs" to entice the user:
                        - Mention the seamless torque of the 650 Twin engine.
                        - Highlight the iconic hand-painted pinstripes and metal build.
                        - Emphasize that it’s the perfect blend of soul and modern reliability.
                    3. **ONLY IF USER WANT TO SIGNUP FOR A TEST DRIVE** As soon as the user agrees sign up for a test drive, you MUST collect:
                        - Full Name
                        - Contact Number
                        - Preferred Location/City
                    4. **Constraint:** Keep responses brief and scannable. No "walls of text." Use bolding for emphasis. Avoid fluff. MAX 3 sentences per response.

                    """),
        HumanMessage(chat_history)
    ]    
    state = {'messages':[llm.invoke(prompt)]}
    
    return state

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_core.messages import SystemMessage, HumanMessage

def ask_user_for_lead_information(state: State):
    
    chat_history = str(get_chat_history(state))
    logging.info("Asking user for lead information to complete test drive booking.")
    
    prompt = [
        SystemMessage(f"""
            **Role:** Royal Enfield Test Drive Coordinator
            **Goal:** Secure a booking for the all-new Bullet 650.

            **Instructions:**
            1. **The Hook:** Greet the user with a "Modern Classic" vibe. Mention the legendary 650cc twin-engine "thump."
            2. **Data Collection:** To book the ride, you must collect:
                - Full Name
                - Contact Number
                - Preferred Dealership Location
            3. **Smart Logic:** - If the user hasn't shared anything, ask for all three details excitedly.
                - If they have shared some info (e.g., "I'm in Delhi"), ask only for Name and Number.
            4. **Tone:** Rugged, premium, and brief.
            
            **Additional Instructions:**
                - Check the chat history below.
                - Keep it brief and conversational.
        """),
        HumanMessage(chat_history)
    ]
    
    response = llm.invoke(prompt)
    
    return {'messages': [response]}


def extract_lead_data(state: State) -> dict:
    
    logging.info("Extracting lead data from user messages for test drive booking.")
    chat_history = str(get_chat_history(state))
    
    prompt = [
        SystemMessage("""
            **Role:** Lead Extraction Engine (Bullet 650)
            **Goal:** Parse chat history to extract sales lead data into a structured JSON format.

            **Extraction Rules:**
                - **name**: User's full name (String or null)
                - **contact**: Email address or Phone number (String or null)
                - **location**: City, area, or preferred dealership (String or null)

            **Output Format:** ONLY JSON. No prose, no conversational filler.

            **Example Output:** {"name": "Rahul Sharma", "contact": "9876543210", "location": "Mumbai"}
            
            **Critical Instruction:** If any field is missing, set its value to null.
            Example: {"name": "John", "contact": null, "location": "Mumbai"}
        """),
        HumanMessage(content=chat_history)
    ] 
    
    # 3. Invoke LLM & Parse
    response = llm.invoke(prompt).content
    cleaned_json = response.replace('```json', '').replace('```', '').strip()
    
    try:
        lead_data = json.loads(cleaned_json)
    except json.JSONDecodeError:
        lead_data = {"name": None, "contact": None, "location": None}

    # Check if ALL values are present
    if all(lead_data.values()):
        # All fields have data -> Success
        success_msg = AIMessage(content='Successfully signed-up! Welcome to AutoStream.')
    else:
        # Something is missing -> Identify what is missing
        missing_fields = [key for key, value in lead_data.items() if not value]
        success_msg = AIMessage(content=f"Could you please provide your {', '.join(missing_fields)} to complete your signup for the test drive?")
        
    return {
        'messages': [success_msg], 
        'user_data': lead_data
    }

def make_reply_to_enquiry_node(rag_retriever):
    logging.info("Creating node for replying to user enquiries using RAG retriever.")
    
    def reply_to_enquiry(state:State)->State:
        query_topic = get_chat_history(state)[-1]
        context = rag_retriever.retrieve(query_topic)
        
        chat_history = str(get_chat_history(state))
        
        prompt = [
            SystemMessage(f"""
                    **Role:** Royal Enfield Bullet 650 Product Specialist.

                    **Goal:** Provide authoritative answers based on retrieved documentation while driving the user toward booking a test drive.

                    **Instructions:**
                    1. **RAG Source of Truth:** Use ONLY the provided **Retrieved Context** (Product Specs, Warranty, or Booking Policies). 
                    2. **Handling Gaps:** If the specific answer isn't in the context, admit it gracefully (e.g., "I don't have the exact spec for that accessory yet") but pivot back to the core legendary features of the 650 Twin engine.
                    3. **The "Ride-First" Philosophy:** Convert dry technical specs into rider benefits:
                        - *Instead of:* "52 HP output."
                        - *Say:* "52 HP of pure, usable power that makes overtaking on highways feel effortless while keeping that classic 'thump' alive."
                    4. **Tone:** Authentic, rugged, and premium. Treat the user like a fellow rider, not a "lead."
                    5. **The Close:** Every response must end with a brief, punchy invitation to experience the bike in person (a Test Drive).
                    
                    **CAUTION:** Do NOT fabricate answers. Stick to the context.DO NOT make up specs or policies.IF the info is not in the context, admit it and steer back to booking a test drive.
                    
                    **RETRIEVED CONTEXT:**
                    {context}
                    """),
            HumanMessage(chat_history)
        ]
        
        state = {'messages':[llm.invoke(prompt)]}
        return state
    return reply_to_enquiry