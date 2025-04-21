# pip install -U chainlit autogen-agentchat autogen-ext[openai] pyyaml

from typing import List, cast

import chainlit as cl
import yaml
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination


import backoff
import openai



async def user_input_func(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    """Get user input from the UI for the user proxy agent."""
    try:
        response = await cl.AskUserMessage(content=prompt).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    if response:
        return response["output"]  # type: ignore
    else:
        return "User did not provide any input."


async def user_action_func(prompt: str, cancellation_token: CancellationToken | None = None) -> str:
    """Get user action from the UI for the user proxy agent."""
    try:
        response = await cl.AskActionMessage(
            content="Pick an action",
            actions=[
                cl.Action(name="approve", label="Approve", payload={"value": "approve"}),
                cl.Action(name="reject", label="Reject", payload={"value": "reject"}),
            ],
        ).send()
    except TimeoutError:
        return "User did not provide any input within the time limit."
    if response and response.get("payload"):  # type: ignore
        if response.get("payload").get("value") == "approve":  # type: ignore
            return "APPROVE."  # This is the termination condition.
        else:
            return "REJECT."
    else:
        return "User did not provide any input."


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:

    gpt_model = "gpt-4o-mini"
    # gpt_model = "gpt-4.1-mini-2025-04-14"

    model_client = OpenAIChatCompletionClient(model=gpt_model)

    # Create agents
    agents = []

    user = UserProxyAgent(
        name="User", 
        description="The user agent.",
        )
    agents.append(user)
    
    surfer1 = MultimodalWebSurfer(
        name="ActivityAdvisor",
        model_client=model_client, 
        #description="A web surfer agent that search for things to do in the destination. It also need to return the booking fee for each activity.",
        description = '''List popular activities available at the destination and provide an estimated cost range for each activity. 
                        Think step-by-step: identify activities first, then estimate typical costs. If cost information is unavailable, note it clearly. 
                        Example: - Activity: City Sightseeing Tour, Cost: $50-70 - Activity: Museum Visit, Cost: $10-20.''',
        start_page="https://www.tripadvisor.com/", 
    )
    agents.append(surfer1)

    surfer2 = MultimodalWebSurfer(
        name="AccomadationAdvisor",
        model_client=model_client, 
        #description="A web surfer agent that search for accommodation in the destination. It also need to return the booking fee for each accommodation.",
        description = '''Search for accommodation options in the destination and provide the booking fee for each. 
                        Think step-by-step: find several hotel or lodging options, then retrieve and list their booking prices. Include the name and a short description for each accommodation. 
                        Example: - Hotel: Grand Palace Hotel, Booking Fee: $120/night - Hostel: City Center Backpackers, Booking Fee: $35/night.''',

        start_page="https://www.tripadvisor.com/", 
    )
    agents.append(surfer2)

    surfer3 = MultimodalWebSurfer(
        name="FlightsearchAgent",
        model_client=model_client, 
        #description="A web surfer agent that search for flight tickets(or other travel tools) in the destination. It also need to return the booking fee for each flight(or other travel methods).",
        description = '''Search for available flight tickets or other transportation methods to the destination, and provide the booking fee for each option. 
                        Think step-by-step: first find several transportation options, then retrieve their prices. Include departure location, transportation type, and cost. 
                        Example: - Flight: New York to Paris, Airline: Air France, Booking Fee: $650 - Train: London to Paris, Booking Fee: $120.''',
        start_page="https://www.tripadvisor.com/",
    )
    agents.append(surfer3)

    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        description="A helpful assistant agent that prints out the options.",
        # tools=
    )
    agents.append(assistant)

    cond1 = TextMentionTermination("TERMINATE")
    team = MagenticOneGroupChat(agents, model_client=model_client, termination_condition=cond1)

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("team", team)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        # cl.Starter(
        #     label="Poem Writing",
        #     message="Write a poem about the ocean.",
        # ),
        # cl.Starter(
        #     label="Story Writing",
        #     message="Write a story about a detective solving a mystery.",
        # ),
        # cl.Starter(
        #     label="Write Code",
        #     message="Write a function that merge two list of numbers into single sorted list.",
        # ),
        cl.Starter(
            label="Travel Itinerary",
            message="""
            Task: Plan a Travel Itinerary

            You will generate a detailed travel itinerary including the following information:
            - Place of Departure (city, country)
            - Destination
            - Travel Dates (schedule)
            - Accommodation
            - Activities (things to do)
            - Budget

            Use the Chain-of-Thought method by completing each task step-by-step, explicitly reasoning at each step. Provide a few-shot example for guidance.

            Step-by-Step Instruction:

            Step 1: Mandatory Information Collection
            First, ask the user explicitly:
            "Please tell me your place of departure (city and country). This is mandatory information to start planning your itinerary."

            Reasoning: The itinerary cannot be planned without knowing the starting location. Ensure the user provides this information clearly.

            Few-shot Example:
            Agent: Please tell me your place of departure (city and country). This is mandatory information.
            User: Boston, USA

            Step 2: Collect Additional Travel Information (with Recommendations)
            Next, sequentially ask the user for destination, travel dates, and expected budget. When asking each question, always provide 3 recommendation options.

            Reasoning: Providing options helps users make quicker and informed decisions.

            Few-shot Example:
            Agent: Where would you like to travel? Here are three popular recommendations:
            1. San Francisco, USA
            2. New York City, USA
            3. Miami, USA
            User: I'd like to go to San Francisco, USA.

            Agent: Great choice! What dates are you planning for your travel? Here are three recommended timeframes:
            1. May 20 - May 27
            2. June 10 - June 17
            3. July 15 - July 22
            User: June 10 - June 17

            Agent: What's your expected budget for the entire trip? Here are three typical budget ranges:
            1. Economy: $1,000 - $1,500
            2. Moderate: $1,500 - $2,500
            3. Luxury: $2,500+
            User: Moderate, around $2,000

            Step 3: Generate Activities and Accommodations
            Use web search to identify suitable activities at the destination and accommodations based on the user's preferences and budget constraints. Clearly state your search criteria.

            Reasoning: To provide accurate recommendations, you need to align activities and accommodations with the user's budget and interests.

            Few-shot Example:
            Agent Search Criteria:
            - Destination: San Francisco, USA
            - Dates: June 10 - June 17
            - Budget: Moderate (~$2,000)
            - Activities: Popular tourist spots, cultural experiences
            - Accommodations: Mid-range hotels or Airbnb

            Sample Agent Output:
            Activities:
            1. Visit Golden Gate Bridge (Free)
            2. Alcatraz Island Tour ($40)
            3. Exploratorium ($30)

            Accommodation Recommendation:
            1. Holiday Inn Golden Gateway ($180/night)
            2. Airbnb near Fisherman's Wharf ($150/night)

            Step 4: Calculate and Present Budget Clearly
            Calculate the total expected cost based on accommodation, activities, transportation, and daily expenses clearly.

            Reasoning: Users appreciate transparency and a clear breakdown of expenses.

            Few-shot Example:
            Budget Breakdown:
            - Flights (Round-trip): $400
            - Accommodation (7 nights x $150): $1,050
            - Activities: $70
            - Meals and Miscellaneous: $400
            - Total: ~$1,920

            Step 5: Final Travel Plan Summary
            Provide a concise and comprehensive summary of the travel itinerary.

            Reasoning: Summaries help users quickly understand and confirm the overall plan.

            Few-shot Example:
            Summary:
            - Departure: Boston, USA
            - Destination: San Francisco, USA
            - Dates: June 10 - June 17
            - Accommodation: Airbnb near Fisherman's Wharf
            - Activities: Golden Gate Bridge, Alcatraz Tour, Exploratorium
            - Total Budget: ~$1,920
            """
        ),
    ]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the team from the user session.
    team = cast(MagenticOneGroupChat, cl.user_session.get("team"))  # type: ignore
    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.

    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        # task=task, 
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content="", author=msg.source)
            # await streaming_response.stream_token(msg.content)
            await cl.Message(
                content=streaming_response.content + msg.content,
                author=msg.source,
            ).send()
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
            await streaming_response.send()
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            # Send the task termination message.
            final_message = "Task terminated. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            # pass
            await cl.Message(content=msg.content, author=msg.source).send()
