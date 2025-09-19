import asyncio
import logging
import os
from videosdk.agents import (AgentSession, CascadingPipeline, ConversationFlow, WorkerJob,
    JobContext, RoomOptions, ChatContext, ChatRole
)
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from backend.controller.qdrant import VectorStore
from backend.config import OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API
from backend.components.agent import VoiceAgent
from backend.components.stt import DeepgramSTT
from backend.components.rag_llm import RAGLLM
logging.basicConfig(level=logging.INFO)
vectorstore = VectorStore()


# Entrypoint
async def entrypoint(ctx: JobContext):
    agent = VoiceAgent()
    llm = RAGLLM(api_key=OPENAI_API_KEY, vectorstore=vectorstore)
    stt = DeepgramSTT(api_key=DEEPGRAM_API_KEY)

    pipeline = CascadingPipeline(
        stt=stt,
        llm=llm,
        tts=ElevenLabsTTS(api_key=ELEVENLABS_API),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.5)
    )
    pipeline.set_agent(agent)
    conversation_flow = ConversationFlow(agent, stt=pipeline.stt, llm=pipeline.llm, tts=pipeline.tts)
    conversation_flow.chat_context = ChatContext()
    pipeline.set_conversation_flow(conversation_flow)
    stt.pipeline = pipeline
    
    async def _process_text_input(self, text: str):
        if not hasattr(self, "chat_context") or self.chat_context is None:
            self.chat_context = ChatContext()
            logging.warning("Created missing chat_context in conversation_flow!")

        # Add user message
        self.chat_context.add_message(ChatRole.USER, text)
        logging.info(f"user input speech: {text}")

        # Run LLM
        if self.llm:
            async for resp in self.llm.chat(self.chat_context):
                if resp.content:
                    # Speak response
                    if self.tts and hasattr(self.tts, "say"):
                        await self.tts.say(resp.content)
                    # Add assistant message
                    self.chat_context.add_message(ChatRole.ASSISTANT, resp.content)

    conversation_flow.process_text_input = _process_text_input.__get__(conversation_flow)

    session = AgentSession(agent=agent, pipeline=pipeline, conversation_flow=conversation_flow)

    async def cleanup_session():
        if session:
            await session.close()

    ctx.add_shutdown_callback(cleanup_session)

    try:
        await ctx.connect()
        logging.info("Waiting for participant...")
        await ctx.room.wait_for_participant()
        logging.info("Participant joined. Starting session...")
        await session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully...")
    finally:
        await cleanup_session()
        await ctx.shutdown()

# Job Context Factory
def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id=os.getenv("ROOM_ID"),
        auth_token=os.getenv("AUTH_TOKEN"),
        name="RAG Voice Agent",
        playground=True
    )
    return JobContext(room_options=room_options)

# Run
if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context())
    job.start()
