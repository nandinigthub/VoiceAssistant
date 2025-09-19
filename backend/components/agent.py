import logging
from videosdk.agents import Agent,ChatContext
from backend.controller.qdrant import VectorStore

logging.basicConfig(level=logging.INFO)
vectorstore = VectorStore()

# Voice Agent
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant that can answer questions using documents.")
        self.chat_context = ChatContext()
        self.vectorstore = vectorstore

    async def on_enter(self) -> None:
        if self.session:
            await self.session.say("Hello! How can I help you today?")

    async def on_exit(self) -> None:
        if self.session:
            await self.session.say("Goodbye!")
