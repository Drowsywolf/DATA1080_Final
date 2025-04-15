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
    # Load model configuration and create the model client.
    with open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    # model_client = ChatCompletionClient.load_component(model_config)
    gpt_model = "gpt-4o-mini"
    # gpt_model = "gpt-4.1-mini-2025-04-14"
    model_client = OpenAIChatCompletionClient(model=gpt_model, max_tokens=300)

    # # Create the assistant agent.
    # assistant = AssistantAgent(
    #     name="assistant",
    #     model_client=model_client,
    #     system_message="You are a helpful assistant.",
    #     model_client_stream=True,  # Enable model client streaming.
    # )

    # # Create the critic agent.
    # critic = AssistantAgent(
    #     name="critic",
    #     model_client=model_client,
    #     system_message="You are a critic. Provide constructive feedback. "
    #     "Respond with 'APPROVE' if your feedback has been addressed.",
    #     model_client_stream=True,  # Enable model client streaming.
    # )

    # # Create the user proxy agent.
    # user = UserProxyAgent(
    #     name="user",
    #     # input_func=user_input_func, # Uncomment this line to use user input as text.
    #     input_func=user_action_func,  # Uncomment this line to use user input as action.
    # )

    # # Termination condition.
    # termination = TextMentionTermination("APPROVE", sources=["user"])

    # # Chain the assistant, critic and user agents using RoundRobinGroupChat.
    # group_chat = RoundRobinGroupChat([assistant, critic, user], termination_condition=termination)

    agents = []

    user = UserProxyAgent(
        name="User", 
        description="The user agent.",
        input_func=user_input_func,  # Use the user input function.
        )
    agents.append(user)
    
    surfer1 = MultimodalWebSurfer(
        name="WebSurfer1",
        model_client=model_client, 
        description="A web surfer agent that search for things to do in the destination. It also need to return the booking fee for each activity.",
        start_page="https://www.tripadvisor.com/", 
    )
    agents.append(surfer1)

    surfer2 = MultimodalWebSurfer(
        name="WebSurfer2",
        model_client=model_client, 
        description="A web surfer agent that search for accommodation in the destination. It also need to return the booking fee for each accommodation.",
        start_page="https://www.tripadvisor.com/", 
    )
    agents.append(surfer2)

    surfer3 = MultimodalWebSurfer(
        name="WebSurfer3",
        model_client=model_client, 
        description="A web surfer agent that search for flight tickets(or other travel tools) in the destination. It also need to return the booking fee for each flight(or other travel methods).",
        start_page="https://www.tripadvisor.com/",
    )
    agents.append(surfer3)

    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        description="A helpful assistant agent that prints out the options.",
        # tools=
    )
    # agents.append(assistant)

    cond1 = TextMentionTermination("TERMINATE")
    group_chat = MagenticOneGroupChat(agents, model_client=model_client, termination_condition=cond1)

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("team", group_chat)  # type: ignore


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
            message="Plan a travel itinerary. " \
            "The final plan should include the following information: place of departure(from which city, country), destination, travel dates(schedule), accommodation, activities(things to do), and budget. " \
            "First, ask the user for place of departure(from which city, country). This is mendatory infomation the user need to answer, so you need to keep asking until the user answers this. " \
            "Second, ask the user for destination, travel dates(schedule) and expected budget. Ask the usr for these one by one, and give 3 recommendation options along with your ask. " \
            # "Accommodation, activities(things to do), the information you need to provide for the user. " \
            # "You need to search for the activities, accommodations and flight(or other travel tools) separately, and the calculate the budget accordingly." \
            "Finally, you need to provide a summary of the travel plan. "
        ),
    ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the team from the user session.
    team = cast(RoundRobinGroupChat, cl.user_session.get("team"))  # type: ignore
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
