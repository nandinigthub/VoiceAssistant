import logging
from typing import Any
import openai
from videosdk.agents import ChatContext, ChatRole, LLMResponse
from backend.controller.qdrant import VectorStore
from videosdk.agents.llm import LLM as BaseLLM


logging.basicConfig(level=logging.INFO)
vectorstore = VectorStore()


# Custom RAG LLM

class RAGLLM(BaseLLM):
    def __init__(self, api_key: str, vectorstore: VectorStore, model: str = "gpt-4"):
        super().__init__()
        self.api_key = api_key
        self.vectorstore = vectorstore
        self.model = model

    async def chat(self, messages: ChatContext, **kwargs) -> Any:
        try:
            msg_list = getattr(messages, "messages", [])
            if not msg_list:
                logging.warning("No user message in chat context")
                return

            last_msg = msg_list[-1]
            last_user_msg = last_msg.content

            # Retrieve RAG context from vector store
            results = self.vectorstore.search(last_user_msg, top_k=3)
            if len(results)>0:
                context_text = "\n".join([r["text"] for r in results])
                prompt = f"User query is asking this question: {last_user_msg} and here is little context about it: {context_text}"
            else:
                prompt = f"User query is asking this question: {last_user_msg}"
                

            augmented_messages = [
                {"role": msg.role.value, "content": prompt if msg == last_msg else msg.content}
                for msg in msg_list
            ]

            if not augmented_messages:
                augmented_messages = [
                    {"role": "system", "content": "You are a helpful assistant, answer user's query."},
                    {"role": "user", "content": prompt}
                ]

            response = openai.chat.completions.create(
                model=self.model,
                messages=augmented_messages,
                **kwargs
            )
            full_content = response.choices[0].message.content
            yield LLMResponse(content=full_content, role=ChatRole.ASSISTANT)

        except Exception as e:
            self.emit("error", f"RAG LLM streaming failed: {e}")

    async def aclose(self):
        pass
