import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import re
import json

load_dotenv()


class Hint(BaseModel):
    """Structured hint output."""
    level: int = Field(..., description="1=light nudge, higher=more direct")
    text: str


class PuzzleHintEngine:
    """Produce hints without giving away the answer at low difficulty."""

    def __init__(self):
        self.system_prompt = (
            "You provide puzzle hints in progressive layers, never spoiling unless difficulty is very low. "
            "At difficulty=1, hints can be direct; at higher numbers, hints should be more vague and encouraging, never revealing the answer."
        )
        self.user_prompt = (
            "Puzzle: {puzzle}\nAttempt: {attempt}\nDifficulty: {difficulty}\n"
            "Return 2-3 hints from gentle to direct. Each hint should be JSON with keys 'level' and 'text'."
        )
        system_template = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_template = HumanMessagePromptTemplate.from_template(self.user_prompt)
        self.prompt = ChatPromptTemplate.from_messages([system_template, human_template])

    def get_hints(self, puzzle: str, attempt: str, difficulty: int = 3) -> List[Hint]:
        temp_map = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9, 5: 1.1}
        temperature = temp_map.get(difficulty, 0.7)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature)

      
        chain = self.prompt | llm

        output_message = chain.invoke({
            "puzzle": puzzle,
            "attempt": attempt,
            "difficulty": difficulty
        })

        raw_text = output_message.content

        
        clean_text = re.sub(r"``````", "", raw_text, flags=re.IGNORECASE).strip()

        try:
            result = json.loads(clean_text)
            if isinstance(result, dict):
                result = [result]
            hints = [Hint(**h) for h in result]
            return hints
        except Exception as e:
            print("Error parsing hints:", e)
            print("Output was:", raw_text)
            return []


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
        return

    engine = PuzzleHintEngine()
    try:
        print("\n🧩 Puzzle Hint Engine — demo\n" + "-" * 40)
        hints = engine.get_hints(
            "I speak without a mouth and hear without ears.",
            attempt="Is it wind?",
            difficulty=2,
        )
        for h in hints:
            print(f"[{h.level}] {h.text}")
    except Exception as e:
        print("Error in demo:", e)


if __name__ == "__main__":
    _demo()
