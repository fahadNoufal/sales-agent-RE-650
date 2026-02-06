from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from src.custom_logger import logging

class UserData(TypedDict):
    # schema for structured lead information
    name: Optional[str]
    contact: Optional[str]
    location: Optional[str]
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_intent: Literal['lead', 'inquiry', 'greeting', 'extract']
    user_data: UserData

class LeadValidationModel(BaseModel):
    """The schema the LLM must follow"""
    name: Optional[str] = Field(None, description="User's full name")
    contact: Optional[str] = Field(None, description="Phone or Email")
    location: Optional[str] = Field(None, description="City/Dealer location")


def get_chat_history(state: State):
    return ['human: '+s.content if isinstance(s, HumanMessage) else 'bot: '+s.content for s in state['messages']]

def route_based_on_intent(state: State):
    intent_raw = state['user_intent'].content.lower().strip()
    
    logging.info(f"Routing based on intent: {intent_raw}")
    
    if 'high-intent' in intent_raw: return 'lead'
    if 'product' in intent_raw or 'inquiry' in intent_raw: return 'inquiry'
    if 'extract' in intent_raw: return 'extract'
    return 'greeting'   