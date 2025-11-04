"""
Assignment 6: Reply Macro Composer — Runtime Configs

Goal: Compose short, consistent reply macros from a customer message and context.

Implement bodies according to docstrings. Prefer small, composable helpers.
Use runtime configs (`.bind`, `.with_config`) to adjust tone and length.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser


class StrOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text


class MacroComposer:
    """Compose reply macros with configurable tone and length."""

    def __init__(self):
        """Initialize any state; prepare prompt strings.

        Requirements:
        - Define a `system_prompt` string describing style (polite, frictionless, concise).
        - Define a `user_prompt` string with variables: {message}, {context}, {style_hint}.
        - Do not build ChatPromptTemplate here; keep only strings and TODOs.
        """
        self.system_prompt = "You craft helpful, concise support macros that sound friendly and professional."
        self.user_prompt = (
            "Customer message:\n{message}\n\nContext:\n{context}\n\nStyle hint: {style_hint}\n"
            "Return a ready-to-send macro with greeting and sign-off."
        )
        # Create ChatPromptTemplate using the above strings and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(self.user_prompt),
        ])

        # Create a base ChatOpenAI LLM (low temperature). Store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Prepare output parser
        self.parser = StrOutputParser()

    def compose_macro(
        self, message: str, context: str, style_hint: str = "neutral"
    ) -> str:
        """Return a polished macro.

        Implement:
        - Bind runtime parameters (e.g., max_tokens, temperature) via `.bind` or `.with_config`.
        - Connect `self.prompt | self.llm | StrOutputParser()`.
        - Invoke with `{"message": message, "context": context, "style_hint": style_hint}`.
        - Return the string content.
        """
        # Bind runtime parameters to llm for this call (example: max_tokens=200)
        llm_configured = self.llm.with_config(max_tokens=200, temperature=0.3)

        # Compose the chain: prompt -> LLM -> parser
        chain = self.prompt | llm_configured | self.parser

        # Call the chain with inputs
        output = chain.invoke({"message": message, "context": context, "style_hint": style_hint})

        return output

    def compose_bulk(
        self, items: List[Dict[str, str]], style_hint: str = "neutral"
    ) -> List[str]:
        """Batch-compose macros for many items.

        Implement:
        - Use the same chain as `compose_macro` but with `.batch` for parallelism.
        - Each item has keys: message, context.
        - Return list of strings in same order.
        """
        # Bind runtime parameters to llm for batch calls
        llm_configured = self.llm.with_config(max_tokens=200, temperature=0.3)

        # Compose the chain: prompt -> LLM -> parser
        chain = self.prompt | llm_configured | self.parser

        # Prepare input list for batch call
        inputs = [
            {"message": item["message"], "context": item["context"], "style_hint": style_hint}
            for item in items
        ]

        # Call batch method
        outputs = chain.batch(inputs)

        return outputs


def _demo():
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mc = MacroComposer()
    try:
        print("\n✉️ Macro Composer — demo\n" + "-" * 40)
        print(
            mc.compose_macro(
                "My package arrived damaged. What can I do?",
                context="Order #123, policy: refund or replacement within 30 days.",
                style_hint="warm",
            )
        )
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()