from langchain.tools import Tool, StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import random
import os
from dotenv import load_dotenv

load_dotenv()


# ======================
# TOOL FUNCTIONS
# ======================

# Single-parameter function (can use regular Tool)
def search_flights(destination: str) -> str:
    """Search for flight prices to destination"""
    prices = [850, 920, 1150, 780, 1320]
    price = random.choice(prices)
    return f"Found flights to {destination} starting from ${price}"


# Multi-parameter functions (need StructuredTool)
def get_weather_forecast(location: str, month: str) -> str:
    """Get weather forecast for location in specific month"""
    weather_patterns = {
        "japan": {
            "march": "Cherry blossom season! 15-20°C, occasional rain",
            "july": "Hot and humid, 30-35°C, rainy season",
            "october": "Perfect weather! 20-25°C, clear skies",
            "december": "Cold, 5-10°C, possible snow"
        }
    }

    location_lower = location.lower()
    month_lower = month.lower()

    if location_lower in weather_patterns and month_lower in weather_patterns[location_lower]:
        return weather_patterns[location_lower][month_lower]
    else:
        return f"Weather data not available for {location} in {month}"


def check_visa_requirements(destination: str, passport: str) -> str:
    """Check visa requirements for travel"""
    visa_info = {
        "japan": {
            "us": "90-day tourist visa waiver available",
            "uk": "90-day tourist visa waiver available",
            "india": "Visa required - apply online 30 days before travel"
        }
    }

    dest_lower = destination.lower()
    passport_lower = passport.lower()

    if dest_lower in visa_info and passport_lower in visa_info[dest_lower]:
        return visa_info[dest_lower][passport_lower]
    else:
        return f"Visa information not available for {passport} passport holders traveling to {destination}"


def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency amounts"""
    rates = {
        "usd_to_jpy": 150.0,
        "jpy_to_usd": 0.0067,
        "eur_to_jpy": 165.0
    }

    rate_key = f"{from_currency.lower()}_to_{to_currency.lower()}"
    if rate_key in rates:
        converted = amount * rates[rate_key]
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
    else:
        return f"Exchange rate not available for {from_currency} to {to_currency}"


# ======================
# PYDANTIC SCHEMAS
# ======================

class WeatherInput(BaseModel):
    location: str = Field(description="The destination/location name")
    month: str = Field(description="The month to check weather for")


class VisaInput(BaseModel):
    destination: str = Field(description="The destination country")
    passport: str = Field(description="The passport country (e.g., 'US', 'UK', 'India')")


class CurrencyInput(BaseModel):
    amount: float = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code (e.g., 'USD', 'EUR')")
    to_currency: str = Field(description="Target currency code (e.g., 'JPY', 'USD')")


# ======================
# CREATE TOOLS
# ======================

# Single-parameter tool (regular Tool)
flight_search_tool = Tool(
    name="search_flights",
    description="Search for flight prices to any destination. Provide the destination name.",
    func=search_flights
)

# Multi-parameter tools (StructuredTool)
weather_tool = StructuredTool(
    name="weather_forecast",
    description="Get weather information for a location in specific month. Use for weather planning and packing advice.",
    func=get_weather_forecast,
    args_schema=WeatherInput
)

visa_tool = StructuredTool(
    name="visa_requirements",
    description="Check visa requirements for international travel. Use when user asks about visas, passports, or entry requirements.",
    func=check_visa_requirements,
    args_schema=VisaInput
)

currency_tool = StructuredTool(
    name="currency_converter",
    description="Convert between different currencies. Use when user asks about costs, budgeting, or money exchange.",
    func=currency_converter,
    args_schema=CurrencyInput
)

travel_tools = [flight_search_tool, weather_tool, visa_tool, currency_tool]

# ======================
# CREATE AGENT
# ======================

travel_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are TravelGPT, an expert travel planning assistant.

Your goal: Help users plan amazing trips by using the right tools in the right order.

Tool Selection Strategy:
1. For trip planning: Start with destination research (weather, visa)
2. For costs: Use flights + currency conversion  
3. For timing: Check weather patterns for best travel months
4. Always provide actionable, specific advice

Available tools:
- search_flights: Takes destination (single parameter)
- weather_forecast: Takes location and month (two parameters)
- visa_requirements: Takes destination and passport country (two parameters)  
- currency_converter: Takes amount, from_currency, to_currency (three parameters)

Be conversational, enthusiastic, and helpful. When using multiple tools, 
explain your reasoning and connect the information logically."""),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create the travel agent
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

travel_agent = create_tool_calling_agent(
    llm=llm,
    tools=travel_tools,
    prompt=travel_agent_prompt
)

travel_executor = AgentExecutor(
    agent=travel_agent,
    tools=travel_tools,
    verbose=True,
    handle_parsing_errors=True
)

# ======================
# TEST THE AGENT
# ======================

print("=== COMPREHENSIVE TRAVEL PLANNING ===")
result = travel_executor.invoke({
    "input": "I want to visit Japan in October. I'm from the US and have a budget of $2000. What should I know?"
})
print(result["output"])

print("\n=== SPECIFIC QUERIES ===")
result = travel_executor.invoke({
    "input": "What's the weather like in Japan in March and what are the visa requirements for US citizens?"
})
print(result["output"])

print("\n=== BUDGET PLANNING ===")
result = travel_executor.invoke({
    "input": "Convert $1500 to Japanese Yen and find flights to Tokyo"
})
print(result["output"])
