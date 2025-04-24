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

# Tools
## tool to search for attactions
def search_top_sights(destination: str):
    """Returns popular tourist attractions at a given destination using SerpAPI's Top Sights API."""
    params = {
        "engine": "google",
        "q": f"Top sights in {destination}",
        "api_key": serp_api_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    top_sights = results["top_sights"]
    return top_sights

## tool to search for hotels
def search_hotels(destination: str, check_in_date: str, check_out_date: str):
    """Returns popular hotels at a given destination using SerpAPI's Top Sights API."""
    params = {
    "engine": "google_hotels",
    "q": f"{destination} hotels",
    "check_in_date": check_in_date,
    "check_out_date": check_out_date,
    "currency": "USD",
    "gl": "us",
    "hl": "en",
    "api_key": serp_api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

## tool to search for flights
def search_flights(departure: str, destination: str, depart_date: str, return_date: str):
    """Returns flights from departure point to destination SerpAPI's Top Sights API."""
    params = {
    "engine": "google_flights",
    "departure_id": departure,
    "arrival_id": destination,
    "outbound_date": depart_date,
    "return_date": return_date,
    "currency": "USD",
    "hl": "en",
    "api_key": serp_api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

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
    Now, you must use the attraction agent to produce a list of attractions.
    You must wait for user's approval of the list before proceeding to next steps.
    Few-shot Example:
    Agent: Here are three popular recommendations, would you like to make any changes?
    1. Eiffle Tower, rating: 4.9, price: $10,
    2. The Louvre, rating: 4.9, price: $10,
    3. Triumph Arc, rating: 4.9, price: free,
    User: I'd like to go to the first two. Then, given the approved options, use the activity booking agent to find appropriate booking links for each attraction.
    Few-shot Example:
    Agent: Here are the booking links for these attractions:
    1. Eiffle Tower, rating: 4.9, price: $10, book at "https://www.getyourguide.com/-l2600/?cmp=ga&cq_src=google_ads&cq_cmp=22393335569&cq_con=180057014769&cq_term=eiffel%20tower%20tickets"
    2. The Louvre, rating: 4.9, price: $10, book at "https://www.louvre.fr/en"


    Then, using the destination, ask the hotel finder to find appropriate accomodations and
    flight finder for flights .


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
    1. Visit Golden Gate Bridge (Free), no need to book.
    2. Alcatraz Island Tour ($40), book at [link]
    3. Exploratorium ($30), book at [link]

    Accommodation Recommendation:
    1. Holiday Inn Golden Gateway ($180/night), book at [link,]
    2. Airbnb near Fisherman's Wharf ($150/night) book at [link]

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
    Provide a concise and comprehensive summary of the travel itinerary.

    Reasoning: Summaries help users quickly understand and confirm the overall plan.

    Few-shot Example:
    Summary:
    - Departure: Boston, USA
    - Destination: San Francisco, USA
    - Dates: June 10 - June 17
    - Accommodation: Airbnb near Fisherman’s Wharf
    - Activities: Golden Gate Bridge, Alcatraz Tour, Exploratorium (with links attached)
    - Total Budget: ~$1,920

    Step 6:
    Make the final, detailed itinerary and shows flight dates, hotel check-ins, and activity planned for each day
    Show final itinerary and get final user approval.
    Few-shot Example:
    Agent: Here is the final itinerary:
    - **Departure:** Boston, USA
- **Destination:** Los Angeles, USA
- **Travel Dates:** May 20 - May 24, 2025
- **Accommodation:** Vibrant Studio Retreat Near DTLA Attractions
  - Price: $516 for 4 nights
  - Check-in: May 20, Check-out: May 25
  - [View Accommodation](https://www.expedia.com/Hotel.h114487924.Hotel-Information?mdpcid=META.HPA.WEB-ORGANIC.VR)

### Activities:
1. **Griffith Observatory** ("https://griffithobservatory.org")
   - Date: May 21
   - Time: 10:00 AM - 12:00 PM
   - **Cost:** Free
2. **Hollywood Sign**
   - Date: May 21
   - Time: 1:00 PM - 3:00 PM
   - **Cost:** Free
3. **Universal Studios Hollywood** ("https://www.universalstudioshollywood.com/web/en/us/theme-park-ticket-deals?")
   - Time: 9:00 AM - 5:00 PM
   - **Cost:** $109

### Budget Breakdown:
- **Flights (Round-trip):** $191
- **Accommodation:** $516 (for 5 nights)
- **Activities:** $137 ($109 for Universal Studios + $28 for LACMA)

#### **Total Estimated Cost:**
- Total: $844
- **Remaining Budget:** $1,156 (from your budget of $2,000)

### Daily Itinerary:
- **Day 1 (May 20):** Arrival in LA, check into accommodation, enjoy dinner nearby.
- **Day 2 (May 21):** Visit Griffith Observatory and the Hollywood Sign.
- **Day 3 (May 22):** Full day of fun at Universal Studios Hollywood.
- **Day 4 (May 22):** Last-minute activities or departure preparations.
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