from langchain.tools import Tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Single-parameter function (can use regular Tool)
def get_weather(location: str) -> str:
    """Get current weather for a location"""
    weather_data = {
        "tokyo": "22°C, partly cloudy, 60% humidity",
        "new york": "18°C, rainy, 75% humidity",
        "london": "15°C, foggy, 80% humidity"
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Current weather in {location}: {weather_data[location_lower]}"
    else:
        return f"Weather data not available for {location}"


# Multi-parameter function (needs StructuredTool)
def calculate_tip(bill_amount: float, tip_percentage: float) -> str:
    """Calculate tip amount and total bill"""
    try:
        tip = bill_amount * (tip_percentage / 100)
        total = bill_amount + tip
        return f"Bill: ${bill_amount:.2f}, Tip ({tip_percentage}%): ${tip:.2f}, Total: ${total:.2f}"
    except Exception as e:
        return f"Error calculating tip: {str(e)}"


# Define input schema for multi-parameter tool
class TipCalculatorInput(BaseModel):
    bill_amount: float = Field(description="The bill amount in dollars")
    tip_percentage: float = Field(description="The tip percentage (e.g., 18 for 18%)")


# Create tools - Note the difference!
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for any location. Pass the location name as a string.",
    func=get_weather
)

# Use StructuredTool for multi-parameter functions
tip_calculator = StructuredTool(
    name="calculate_tip",
    description="Calculate tip and total amount for a restaurant bill. Provide the bill amount and tip percentage.",
    func=calculate_tip,
    args_schema=TipCalculatorInput
)

# Create the agent
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

tools = [weather_tool, tip_calculator]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.

Available tools:
- get_weather: Takes one parameter (location)
- calculate_tip: Takes two parameters (bill_amount and tip_percentage)

Use tools when you need external information or calculations.
Be conversational and helpful in your responses."""),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent and executor
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Test the corrected agent
print("=== WEATHER QUERY ===")
result = agent_executor.invoke({"input": "What's the weather like in Tokyo?"})
print(result["output"])

print("\n=== TIP CALCULATION ===")
result = agent_executor.invoke({"input": "I have a $85 dinner bill and want to tip 18%. What's the total?"})
print(result["output"])

print("\n=== COMPLEX QUERY ===")
result = agent_executor.invoke({"input": "Check the weather in London and help me calculate a 20% tip for a $150 bill"})
print(result["output"])
