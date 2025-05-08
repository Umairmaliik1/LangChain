import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
# === Global flag ===
should_close = False

# === Tool ===
@tool
def respond_to_greeting(input: str) -> str:
    """Responds to the user with a greeting or general inquiry."""
    if "hi" in input.lower() or "hello" in input.lower():
        return "Hi! Please enter your name,email and age."
    return "Please enter your name,email and age."

@tool
def ask_info(input: str) -> str:
    """Extracts name, email, and age from the user input."""
    # Simple parse
    return f"Received info: {input}"
@tool
def summarize_and_confirm(user_info: str) -> str:
    """
    Summarizes the user's information from a string and asks for confirmation.

    Args:
    - user_info (str): A string containing the user's name, email, and age.

    Returns:
    - str: A confirmation prompt asking the user to confirm their details.
    """
    # Assuming the user_info string is formatted as: "Name: John Doe, Email: john.doe@example.com, Age: 28"
    
    # Parse the user_info string to extract name, email, and age
    try:
        user_info_dict = {info.split(":")[0].strip(): info.split(":")[1].strip() for info in user_info.split(",")}
        name = user_info_dict.get("Name")
        email = user_info_dict.get("Email")
        age = user_info_dict.get("Age")
    except Exception as e:
        return f"Error parsing user info: {e}"

    # Create a summary message
    summary_message = (
        f"Here is the information I have received:\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Age: {age}\n\n"
        "Please confirm if these details are correct. Type 'yes' to confirm or 'no' to provide new information."
    )

    return summary_message

@tool
def close_chat(input: str) -> str:
    """Ends the chat if the user confirms."""
    global should_close
    should_close = True
    return "Thank you! Closing the chat now."

# === LLM Setup ===
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.7,
    max_retries=2,
)

# === Wrap with tool calling ===
llm_with_tools = llm.bind_tools([respond_to_greeting,ask_info,summarize_and_confirm,close_chat])

# === Memory ===
memory = ConversationBufferMemory(return_messages=True)
memory.clear()
# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a friendly assistant that always uses the provided tools to interact with the user.
When the user says hi or greets, use the respond_to_greeting tool.
After greeting, use the ask_info tool to collect the user's name, email, and age.
Once the information is collected, use the summarize_and_confirm tool to summarize the details and ask the user to confirm.
If the user confirms the details, use the close_chat tool to end the conversation.
Do not answer on your own; always use a tool.

For every step, follow this format:
Thought: [Your reasoning here]
Action: [The tool you want to use]
Action Input: [The input for the tool]

If you cannot determine the next step, explain your reasoning in the Thought section and stop."""), 
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
    
])


# === Agent Setup ===
agent = initialize_agent(
    tools=[respond_to_greeting, ask_info, summarize_and_confirm, close_chat],
    llm=llm_with_tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    prompt=prompt,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,  # Increase the iteration limit
    max_execution_time=60  # Optional: Set a time limit
)

# === Chat Loop ===


while True:
    user_input = input("You: ")

    response = agent.run(user_input)

    print("Assistant:", response)
    
    if should_close:
        break

