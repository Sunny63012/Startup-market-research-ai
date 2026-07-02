from pathlib import Path
from market_research_crew.Evaluation.evaluator import Evaluator


class Benchmark:

    @staticmethod
    def evaluate_report() -> dict:
        report_path = Path("reports/report.md")

        if not report_path.exists():
            raise FileNotFoundError(
                f"Report not found at {report_path.resolve()}. "
                "Ensure the crew completed successfully and wrote the report."
            )

        report = report_path.read_text(encoding="utf-8")

        accuracy = Evaluator.evaluate_accuracy(report)
        hallucination = Evaluator.evaluate_hallucination(report)

        return {
            "accuracy": accuracy,
            "hallucination": hallucination,
        }