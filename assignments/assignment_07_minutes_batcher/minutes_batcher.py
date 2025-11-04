import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from dotenv import load_dotenv
load_dotenv()

class StrOutputParser(BaseOutputParser):
    """Output parser that returns string as is."""
    def parse(self, text: str) -> str:
        return text

class MinutesBatcher:
    """Summarize transcripts into minutes and action items."""

    def __init__(self):
        self.system_prompt = "You produce crisp meeting minutes and bullet action items with owners and due dates."
        self.user_prompt = (
            "Title: {title}\nTranscript:\n{transcript}\n\n"
            "Return sections: MINUTES (3-5 bullets), ACTIONS (bullets with owner;date)."
        )
        # Prompt Template
        system_template = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_template = HumanMessagePromptTemplate.from_template(self.user_prompt)
        self.prompt = ChatPromptTemplate.from_messages([system_template, human_template])
        # LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        # Chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=StrOutputParser())

    def summarize_one(self, title: str, transcript: str) -> str:
        """Return minutes+actions for a single transcript."""
        result = self.chain.invoke({"title": title, "transcript": transcript})
        return result

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[str]:
        """Return minutes+actions for a batch of transcripts."""
        results = self.chain.batch(items)
        return results

def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mb = MinutesBatcher()
    try:
        print("\n📝 Minutes & Actions — demo\n" + "-" * 40)
        print(
            mb.summarize_one(
                "Sprint Planning",
                "Discussed backlog grooming, two blockers, and deployment window next Tuesday."
            )
        )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    _demo()
