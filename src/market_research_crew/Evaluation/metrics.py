import time


class Metrics:

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.latency = 0
        self.llm_calls = 0
        self.total_tokens = 0
        self.cost = 0
        self.failures = 0
        self.total_runs = 0

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.end_time = time.perf_counter()
        self.latency = round(self.end_time - self.start_time, 2)

    def increment_calls(self):
        self.llm_calls += 1

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens

    def calculate_cost(self):
        # OpenRouter GPT-4o-mini / OSS models ~ $0.002 per 1K tokens (adjust per model)
        COST_PER_1K_TOKENS = 0.002
        self.cost = round((self.total_tokens / 1000) * COST_PER_1K_TOKENS, 4)

    def register_success(self):
        self.total_runs += 1

    def register_failure(self):
        self.total_runs += 1
        self.failures += 1

    @property
    def failure_rate(self):
        if self.total_runs == 0:
            return 0
        return round((self.failures / self.total_runs) * 100, 2)
    