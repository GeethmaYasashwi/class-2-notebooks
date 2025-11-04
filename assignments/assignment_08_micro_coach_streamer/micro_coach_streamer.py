"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain


load_dotenv()
from langchain_core.callbacks import BaseCallbackHandler

class PrintTokens(BaseCallbackHandler):
    """Fully compatible callback for LangChain streaming."""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)


class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders."""
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic and brief."
        )
        self.user_prompt = (
            "Goal: {goal}\nTime: {time_available}\nReturn a 3-step plan."
        )

        # Prompts
        system_template = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_template = HumanMessagePromptTemplate.from_template(self.user_prompt)
        self.stream_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
        self.plain_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

        # Both LLMs use the same params/model
        self.llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
        self.llm_plain = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=False)

        # Chains
        self.stream_chain = LLMChain(llm=self.llm_streaming, prompt=self.stream_prompt)
        self.plain_chain = LLMChain(llm=self.llm_plain, prompt=self.plain_prompt)

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path."""
        user_input = {"goal": goal, "time_available": time_available}
        if stream:
            printer = PrintTokens()
            # Use LangChain's with_config to attach callback
            self.stream_chain.with_config({"callbacks": [printer]}).invoke(user_input)
            print()  # for neatness
            return ""  # streamed output already printed, return empty string or status
        else:
            result = self.plain_chain.invoke(user_input)
            return result


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
