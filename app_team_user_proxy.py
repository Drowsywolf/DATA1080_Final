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

import asyncio
import nest_asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, ExternalTermination
import backoff
import openai

from autogen_core.tools import FunctionTool
from typing_extensions import Annotated

from playwright.async_api import async_playwright
from serpapi import GoogleSearch
import re
import requests
from bs4 import BeautifulSoup
from statistics import median
from typing import List, Dict, Annotated
serp_api_key = "cfdd0170aa9f66b15819113418aac1b17b69b91195c39edfd90ca1183c5afb5f"
gpt_model = "gpt-4o-mini"

def search_booking_info(query: str) -> dict:
    """Search for booking info (activity/hotel/flight) and return name, link, cost, and note."""
    search = GoogleSearch({
        "q": query,
        "api_key": serp_api_key,
        "hl": "en",
        "gl": "us"
    })
    results = search.get_dict()

    booking_info = {
        "activity": query,
        "link": "tripadvisor.com",
    }

    top_link = ""
    for result in results.get("organic_results", []):
        link = result.get("link", "")
        if any(domain in link for domain in [
            "viator.com", "getyourguide.com", "booking.com",
            "expedia.com", "trip.com", "kayak.com", "tripadvisor.com", "bing.com",
            "travelocity.com", "hotels.com", "airbnb.com", "skyscanner.com",
        ]):
            top_link = link
            break

    if not top_link:
        return booking_info

    try:
        booking_info["link"] = top_link
        return booking_info

    except Exception as e:
        booking_info["link"] = "bing.com"
        return booking_info

search_tool = FunctionTool(
    name="search_booking_info",
    func=search_booking_info,
    description="Use this to perform necessary web searches. "
)

async def get_playwright():
    playwright = await async_playwright().start()
    return playwright

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
    model_client = OpenAIChatCompletionClient(model=gpt_model, temperature=0.7)

    # Create agents
    agents = []
    user = UserProxyAgent(
            name="User",
            description="The user agent.",
            input_func=user_input_func,
            )
    agents.append(user)

    # Agent to check the budget
    budget_checker_agent = AssistantAgent(
        name="BudgetController",
        model_client=model_client,
        description="Compare total travel costs to the user-defined budget and say whether it fits.")
    agents.append(budget_checker_agent)

    # Agent to search for attractions
    AttractionLink_agent = AssistantAgent(
        name="AttractionLinkFinder",
        model_client=model_client,
        tools=[search_top_sights],
        description='''Using the search tool, recommend top attractions for a given destination,
        enter departure and return dates in the format of YYYY-MM-DD.
        Example:
        Departure: Boston
        Destination: Paris
        Departure Date: 2025-04-24
        Return Date: 2025-04-29
        Return a number of attractions appropriate to the trip length based on top reviews. Read the json file and return the list with each entry having
        attraction name, description, link, and price.''')
    agents.append(AttractionLink_agent)


    # Agent to search for hotels
    HotelLink_agent = AssistantAgent(
        name="HotelLinkFinder",
        model_client=model_client,
        tools=[search_hotels],
        description='''Using the search tool, return hotels from the destination according to the chosen price range,
        enter departure and return dates in the format of YYYY-MM-DD.
        Example:
        Departure: Boston
        Destination: Paris
        Departure Date: 2025-04-24
        Return Date: 2025-04-29
        Read the json file and
        return the list with each entry having
        hotel name, description, link, and price.''')
    agents.append(HotelLink_agent)


    # Agent to search for flights
    FlightLink_agent = AssistantAgent(
        name="FlightLinkFinder",
        model_client=model_client,
        tools=[search_flights],
        description='''Using the search tool, destinaton and departure must be entered in the format of 3 capital letters
        enter departure and return dates in the format of YYYY-MM-DD.
        Example:
        Departure: BOS
        Destination: AUS
        Departure Date: 2025-04-24
        Return Date: 2025-04-29
        Return flights to and from the destination according to the chosen price range.
        Read the json file and return the list with each entry having
        flight name, description, link, and price.
        Post all links to the orchestrator''')
    agents.append(FlightLink_agent)

    # Agent to schedule the trip
    schedule_maker = AssistantAgent(
            name="ScheduleMaker",
            model_client = model_client,
            description="""You are a schedule maker and have extensive knowledge of travel route and map.
            Based on the activities and destinations, create a daily itinerary:
            1. Optimize routes between locations
            2. Include travel times
            3. Balance activities per day"""
        )
    agents.append(schedule_maker)


    # Create a group chat
    cond1 = TextMentionTermination("TERMINATE")
    external_termination = ExternalTermination()
    team = MagenticOneGroupChat(agents, model_client=model_client, termination_condition=cond1 | external_termination)

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
            message= """
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

    Reasoning: The itinerary cannot be planned without knowing the starting location. Ensure the user provides this information clearly. Do not excute this step after the user gives a feasible answer.

    Few-shot Example:
    Agent: Please tell me your place of departure (city and country). This is mandatory information.
    User: Boston, USA

    Step 2: Collect Additional Travel Information (with Recommendations)
    Next, sequentially ask the user for destination, travel dates, and expected budget. When asking each question, always provide 3 recommendation options.

    Reasoning: Providing options helps users make quicker and informed decisions. Do not excute this step after the user gives a feasible answer.

    Few-shot Example:
    Agent: Where would you like to travel? Here are three popular recommendations:
    1. San Francisco, USA
    2. New York City, USA
    3. Miami, USA
    User: I'd like to go to San Francisco, USA.

    Agent: Great choice! What dates are you planning for your travel? Here are three recommended timeframes:
    1. May 20 - May 22, 2025
    2. June 10 - June 15, 2025
    3. July 15 - July 29, 2025
    User: June 10 - June 11, 2025

    Agent: What's your expected budget for the entire trip? Here are three typical budget ranges:
    1. Economy: $1,000 - $1,500
    2. Moderate: $1,500 - $2,500
    3. Luxury: $2,500+
    User: Moderate, around $2,000

    Step 3: Generate Flights, Activities and Accommodations
    Now, you must use the attraction agent to produce a list of attractions.
    Then, using the destination, ask the hotel finder to find appropriate accomodations and
    flight finder for flights.
    You must wait for user's approval of the list before proceeding to next steps.
    Few-shot Example:
    Agent: Here are three popular recommendations, would you like to make any changes?
    1. Eiffle Tower, rating: 4.9, price: $10
    2. The Louvre, rating: 4.9, price: $10
    3. Triumph Arc, rating: 4.9, price: $10
    User: I'd like to go to the first two.

    Reasoning: To provide accurate recommendations, you need to align activities and accommodations with the user's budget and interests.

    Few-shot Example:
    Agent Search Criteria:
    - Destination: San Francisco, USA
    - Dates: June 10 - June 17
    - Budget: Moderate (~$2,000)
    - Activities: Popular tourist spots, cultural experiences
    - Transportation: Round-trip flights
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
    Once all information is collected, calculate the total expected cost based on accommodation, activities, transportation, and daily expenses clearly.

    Reasoning: Users appreciate transparency and a clear breakdown of expenses.

    Few-shot Example:
    Budget Breakdown:
    - Flights (Round-trip): $400
    - Accommodation (7 nights x $150): $1,050
    - Activities: $70
    - Meals and Miscellaneous: $400
    - Total: ~$1,920

    Step 5: Final Travel Plan Summary
    Provide a concise and comprehensive summary of the travel itinerary, ask if the user approve of the summary, then send to schedule making agent for final itinerary. Remember to include links to the activities, hotels, and flights.

    Reasoning: Summaries help users quickly understand and confirm the overall plan.

    Few-shot Example:
    Agent:
    Here is your travel itinerary summary:
    - Departure: Boston, USA
    - Destination: San Francisco, USA
    - Dates: June 10 - June 17
    - Accommodation: Airbnb near Fisherman's Wharf[Link: ...],
    - Activities: 
            Golden Gate Bridge[Link1: ...], 
            Alcatraz Tour[Link2: ...], 
            Exploratorium[Link3: ...], 
            ...
    - Flights: Round-trip from Boston to San Francisco[Link: ...]
    - Total Budget: ~$1,920
    Would you like to proceed with this plan? (yes/no)
    User: Yes

    Step 6:
    Show final itinerary, and then ask if the user wants any changes.
    Reasoning - the final result should align with the user's preferences
    """
        ),
    ]


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