class AgentState:
    task: str
    step_results: dict
    artifacts: dict
    memory: list
    current_step: int
    execution_trace: list

class Agent:
    def __init__(self, name, description, skill):
        self.name = name
        self.description = description
        self.skill = skill
        self.state = AgentState()

    def execute_task(self, task):
        self.state.task = task
        self.state.step_results = {}
        self.state.artifacts = {}
        self.state.memory = []
        self.state.current_step = 0
        self.state.execution_trace = []

        for step in self.skill.steps:
            result = step.execute(self.state)
            self.state.step_results[step.name] = result
            self.state.execution_trace.append((step.name, result))
            if step.should_stop(result):
                break