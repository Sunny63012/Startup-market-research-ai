from crewai import LLM
import os
import re


llm = LLM(
    model="openrouter/openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def _parse_number(response) -> int:
    """Extract first integer found in LLM response string."""
    text = str(response).strip()
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return 0


class Evaluator:

    @staticmethod
    def evaluate_accuracy(report) -> int:
        prompt = f"""
        You are an expert startup evaluator.

        Evaluate this report.

        Score:

        Market Research: /20
        Competitor Analysis: /20
        Customer Insights: /20
        Product Strategy: /20
        Business Analysis: /20

        Return ONLY a number from 0-100. No explanation. No text. Just the number.

        REPORT:
        {report}
        """
        response = llm.call(prompt)
        return _parse_number(response)

    @staticmethod
    def evaluate_hallucination(report) -> int:
        prompt = f"""
        Check this report for hallucinations.

        Find:

        1. Fake companies
        2. Fake statistics
        3. Unsupported claims
        4. Contradictions

        Return ONLY a number from 0-100 representing the hallucination percentage.
        No explanation. No text. Just the number.

        REPORT:
        {report}
        """
        response = llm.call(prompt)
        return _parse_number(response)