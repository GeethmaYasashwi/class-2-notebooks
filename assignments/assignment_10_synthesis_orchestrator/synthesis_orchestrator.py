"""
Assignment 10: Synthesis Orchestrator (Two-Stage Pipeline)

Goal: Extract key claims from multiple short notes in parallel, then synthesize
them into a single, coherent summary highlighting agreements and conflicts.
"""
# synthesis_orchestrator.py

import os
from typing import List
from dotenv import load_dotenv


load_dotenv()  

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class SynthesisOrchestrator:
    """Two-stage pipeline: extractor (batch) → synthesizer (single).

    Implementations build two chains and wire them together.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - extractor_system / extractor_user (variables: {note})
        - synthesizer_system / synthesizer_user (variables: {claims})
        - placeholders for prompts, llm(s), and chains.
        """
        self.extractor_system = "You extract 1-2 key claims from a note, neutral voice."
        self.extractor_user = "Note: {note}\nReturn bullet points of key claims."

        self.synth_system = "You synthesize claims into a compact, balanced summary."
        self.synth_user = (
            "Claims from multiple notes:\n{claims}\n"
            "Return sections: Overall Summary; Agreements; Conflicts. Keep concise."
        )

        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  

        
        self.extract_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.extractor_system),
                ("user", self.extractor_user),
            ]
        )
        self.extract_chain = self.extract_prompt | self.llm | StrOutputParser()

        
        self.synth_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.synth_system),
                ("user", self.synth_user),
            ]
        )
        self.synth_chain = self.synth_prompt | self.llm | StrOutputParser()

    def extract_claims(self, notes: List[str]) -> List[str]:
        """Return a list of extracted claims strings, one per note.

        Implemented using `.batch()` on the extractor chain.
        """
        if not notes:
            return []
       
        inputs = [{"note": n} for n in notes]
        outputs = self.extract_chain.batch(inputs)  
        
        return [str(o).strip() for o in outputs]

    def synthesize(self, claims: List[str]) -> str:
        """Return a synthesis from already-extracted claims.

        Invoke synthesizer chain with a joined claims string.
        """
        
        joined = "\n\n".join(claims) if claims else ""
        result = self.synth_chain.invoke({"claims": joined})
        return str(result).strip()

    def run(self, notes: List[str]) -> str:
        """End-to-end: extract claims (batch) then synthesize a final output."""
        claims = self.extract_claims(notes)
        return self.synthesize(claims)


def _demo():
   
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.") 
    orch = SynthesisOrchestrator()
    notes = [
        "Team A reduced latency by 20% after switching cache strategy.",
        "Users report fewer timeouts; however, spikes still occur on Mondays.",
        "Data suggests cache hit rate improved but cold-starts remain high.",
    ]
    try:
        print("\n🧪 Synthesis Orchestrator — demo\n" + "-" * 42)
        print(orch.run(notes))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
