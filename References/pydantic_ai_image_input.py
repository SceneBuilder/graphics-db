import httpx

from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from graphics_db_server.core.config import (
    VLM_MODEL_NAME,
    VLM_PROVIDER_BASE_URL,
)

image_response = httpx.get('https://iili.io/3Hs4FMg.png')  # Pydantic logo

client = AsyncOpenAI(base_url=VLM_PROVIDER_BASE_URL, api_key="empty")
model = OpenAIChatModel(
    VLM_MODEL_NAME,
    provider=OpenAIProvider(base_url=VLM_PROVIDER_BASE_URL, api_key="EMPTY"),
)

agent = Agent(model=model)
result = agent.run_sync(
    [
        'What company is this logo from?',
        BinaryContent(data=image_response.content, media_type='image/png'),  
    ]
)
print(result.output)
# > This is the logo for Pydantic, a data validation and settings management library in Python.