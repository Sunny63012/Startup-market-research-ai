from market_research_crew.crew import MarketResearchCrew
from market_research_crew.Evaluation.metrics import Metrics
from market_research_crew.Evaluation.benchmark import Benchmark


def run():
    """
    Run the crew.
    """

    inputs = {
        "product_idea": "An AI powered tool that summarizes youtube videos on my channel and posts the summary on various social media platforms like LinkedIn, Instagram, Facebook, X, WhatsApp"
    }

    metrics = Metrics()

    try:
        # Inject metrics into the crew instance BEFORE calling .crew()
        crew_instance = MarketResearchCrew()
        crew_instance._metrics = metrics

        metrics.start_timer()

        result = crew_instance.crew().kickoff(inputs=inputs)

        metrics.stop_timer()

        metrics.register_success()

        metrics.calculate_cost()

        benchmark = Benchmark.evaluate_report()

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Latency: {metrics.latency} sec")
        print(f"LLM Calls: {metrics.llm_calls}")
        print(f"Tokens: {metrics.total_tokens}")
        print(f"Cost: ${metrics.cost}")
        print(f"Accuracy: {benchmark['accuracy']}")
        print(f"Hallucination: {benchmark['hallucination']}%")
        print(f"Failure Rate: {metrics.failure_rate}%")
        print("=" * 50)

        return result

    except Exception as e:
        metrics.register_failure()
        raise Exception(f"An error occurred while running the crew: {e}")