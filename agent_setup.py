import ast
import operator
import os
from datetime import datetime
import requests
import wikipedia
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()  # Load environment variables from .env file

GPT_OSS_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)


# ---------------------------------------------------------
# TOOL 1: OpenWeatherMap Live Weather Tool
# ---------------------------------------------------------
@tool
def get_current_weather(city: str) -> str:
    """Get the current live weather (temperature, feels-like, humidity, condition, wind)
    for a given city name. Use this whenever the user asks about current weather or temperature.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is not configured."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Could not fetch weather for '{city}': {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"].capitalize()
        wind = data["wind"]["speed"]
        country = data["sys"].get("country", "")

        return (
            f"Weather in {city}, {country}: {desc}. "
            f"Temperature: {temp}°C (Feels like {feels_like}°C), "
            f"Humidity: {humidity}%, Wind Speed: {wind} m/s."
        )
    except Exception as e:
        return f"Weather API request failed: {str(e)}"


# ---------------------------------------------------------
# TOOL 2: Safe Precision Math Calculator
# ---------------------------------------------------------
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_math_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_math_node(node.left), _eval_math_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_math_node(node.operand))
    raise ValueError("Unsupported mathematical expression")


@tool
def calculate_math(expression: str) -> str:
    """Evaluate a mathematical arithmetic expression accurately (e.g., '(245 * 18) / 4' or '2 ** 10').
    Use this for any arithmetic, percentage, or numerical calculation instead of guessing.
    """
    try:
        cleaned = expression.replace("^", "**").strip()
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_math_node(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Math error for '{expression}': {str(e)}"


# ---------------------------------------------------------
# TOOL 3: Wikipedia Concept Lookup
# ---------------------------------------------------------
@tool
def lookup_wikipedia(topic: str) -> str:
    """Look up a concise factual summary from Wikipedia for scientific concepts,
    historical events, famous personalities, or geography.
    """
    try:
        summary = wikipedia.summary(topic, sentences=3, auto_suggest=True)
        return f"Wikipedia summary for '{topic}':\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Topic '{topic}' is ambiguous. Possible matches: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{topic}'."
    except Exception as e:
        return f"Wikipedia lookup error: {str(e)}"


# ---------------------------------------------------------
# BUILD THE LANGCHAIN AGENT (GPT-OSS ONLY)
# ---------------------------------------------------------
def build_classroom_agent(model_name: str | tuple[str, ...] = "openai/gpt-oss-120b"):
    """Creates and returns a compiled LangChain agent graph using GPT-OSS on Groq."""
    if isinstance(model_name, (tuple, list)):
        if len(model_name) != 1:
            raise ValueError(f"Expected a single GPT-OSS model name, got {model_name!r}.")
        model_name = model_name[0]

    model_name = str(model_name).strip()

    if model_name not in GPT_OSS_MODELS:
        raise ValueError(
            f"Only GPT-OSS models are allowed: {', '.join(GPT_OSS_MODELS)}. Got {model_name!r}."
        )

    # Tool 4: Tavily Real-Time Web Search
    tavily_tool = TavilySearch(
        max_results=3,
        topic="general",
        include_answer=True,
    )

    tools = [
        get_current_weather,
        tavily_tool,
        calculate_math,
        lookup_wikipedia,
    ]

    llm = ChatGroq(
        model=model_name,
        temperature=0,  # Deterministic tool routing
    )

    today = datetime.now().strftime("%B %d, %Y")
    system_prompt = (
        f"You are an intelligent, friendly AI Agent demonstrating Agentic AI to Class 12 students. "
        f"Today's date is {today}. "
        "You have access to four tools: live weather, real-time web search (Tavily), a math calculator, "
        "and Wikipedia. Always use the appropriate tool(s) when real-time data, facts, or calculations "
        "are needed. If a question requires multiple steps (like checking weather and doing math on it), "
        "call the tools in sequence. Present your final answer clearly with bullet points so students "
        "can easily read it."
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent