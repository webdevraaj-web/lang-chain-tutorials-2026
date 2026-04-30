from langchain.agents import create_agent
from langchain_community.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# 🚀 FASTAPI INIT
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 🔐 ENV
# -------------------------------
load_dotenv('.env')
GROQ_KEY = os.getenv('GROQ_KEY')

# -------------------------------
# 🔧 PRICING LOGIC
# -------------------------------
def calculate_price(room, people, season_name):
    base_price = int(room["base_price"])
    extra_price = int(room["extra_person_price"])

    season_price = base_price

    for season in room.get("seasons", []):
        if season_name.lower() in season["season_name"].lower():
            season_price = int(season["price"])

    if people <= 2:
        return season_price
    else:
        return season_price + (people - 2) * extra_price


# -------------------------------
# 🏨 TOOL
# -------------------------------
import json
@tool
def get_hotels_with_price(location: str, people: int, season: str):
    """based on the user query find all the paramete in this api and return given location and seasons based hotels only with calculated price and provide hotels only related this 919816440734 number based only"""

    url = "https://silver-spoonbill-286441.hostingersite.com/wp-json/hm/v1/hotels?phone=919816440734"
    data = requests.get(url).json()

    results = []

    for hotel in data.get("hotels", []):
        if location.lower() in hotel["location"].lower():

            for room in hotel.get("rooms", []):
                total_price = calculate_price(room, people, season)

                results.append({
                    "hotel_name": hotel["hotel_name"],
                    "location": hotel["location"],
                    "room_type": room["room_type"],
                    "total_price": total_price
                })


    return json.dumps(results)

# -------------------------------
# 🤖 LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_KEY
)

# -------------------------------
# 🧠 AGENT
# -------------------------------
agent = create_agent(
    model=llm,
    tools=[get_hotels_with_price],
    system_prompt="""
You are a travel assistant.

Flow:
- Ask ONE question at a time:
    1. Travel season
    2. Number of people
    3. Location

Rules:
- rember if requirements not matched said i no data found related your information
- Do NOT assume anything
- When all info is collected → call tool get_hotels_with_price
- Show only provide details related best hotels with total price check the seasons also show season based price too get this by the season date
"""
)

# -------------------------------
# 📩 REQUEST MODEL
# -------------------------------
class ChatRequest(BaseModel):
    messages: list


# -------------------------------
# 🌐 API ENDPOINT
# -------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.invoke({
        "messages": req.messages
    })

    reply = result["messages"][-1].content

    return {"reply": reply}