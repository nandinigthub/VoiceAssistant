import asyncio
import logging
from videosdk.plugins.deepgram import DeepgramSTT as BaseDeepgramSTT
from videosdk.agents import SpeechEventType

logging.basicConfig(level=logging.INFO)


# Custom Deepgram STT
class DeepgramSTT(BaseDeepgramSTT):
    def __init__(self, *args, pipeline=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline

    def _handle_ws_message(self, msg: dict):
        responses = super()._handle_ws_message(msg)

        for r in responses:
            if r.event_type == SpeechEventType.FINAL:
                text = r.data.text
                logging.info(f"transcript: {text}")
                if self.pipeline and hasattr(self.pipeline, "conversation_flow"):
                    # Schedule coroutine correctly
                    asyncio.create_task(
                        self.pipeline.conversation_flow.process_text_input(text)
                    )
        return responses