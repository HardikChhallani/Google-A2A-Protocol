import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

from multi_tool_agent.agent import *

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

print("\nEnvironment configured.")

# Example tool usage (optional self-test)
print(get_weather("New York"))
print(get_weather("Paris"))

# @title Define the Weather Agent
# Use one of the model constants defined earlier
AGENT_MODEL = "gemini-2.0-flash" # Starting with a powerful Gemini model

weather_agent = Agent(
    name="weather_agent_v1",
    model=AGENT_MODEL, # Specifies the underlying LLM
    description="Provides weather information for specific cities.", # Crucial for delegation later
    instruction="You are a helpful weather assistant. Your primary goal is to provide current weather reports. "
                "When the user asks for the weather in a specific city, "
                "you MUST use the 'get_weather' tool to find the information. "
                "Analyze the tool's response: if the status is 'error', inform the user politely about the error message. "
                "If the status is 'success', present the weather 'report' clearly and concisely to the user. "
                "Only use the tool when a city is mentioned for a weather request.",
    tools=[get_weather, get_current_time], # Make the tool available to this agent
)

print(f"\nAgent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")


# @title Setup Session Service and Runner

# --- Session Management ---
# Key Concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"\nSession created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.
runner = Runner(
    agent=weather_agent, # The agent we want to run
    app_name=APP_NAME,   # Associates runs with our app
    session_service=session_service # Uses our session manager
)
print(f"\nRunner created for agent '{runner.agent.name}'.")

# @title Define Agent Interaction Function
import asyncio
from google.genai import types # For creating message Content/Parts

async def call_agent_async(query: str):
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
      # You can uncomment the line below to see *all* events during execution
      # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")


# We need an async function to await our interaction helper
async def run_conversation():
    await call_agent_async("What is the weather like in London?")
    await call_agent_async("How about Paris?") # Expecting the tool's error message
    await call_agent_async("Tell me the weather in New York")
    await call_agent_async("which cities i got information")

if __name__ == '__main__':
    asyncio.run(run_conversation())