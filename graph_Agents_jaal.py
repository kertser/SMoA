import openai
import httpx
import logging
import asyncio
import os
from dotenv import load_dotenv
import networkx as nx
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash_cytoscape import Cytoscape
import dash_bootstrap_components as dbc

# Loading environment variables, like API keys
load_dotenv()

# Add detailed logging configuration
logging.basicConfig(level=logging.INFO)

# Create graph to track agent interactions
interaction_graph = nx.DiGraph()


async def openai_post_request(messages, model_name, max_tokens, temperature):
    """Send a request to the OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3  # Number of attempts
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Check for successful request
            return response.json()  # Return JSON response from OpenAI API
        except Exception as e:
            logging.error(f"Error during OpenAI request: {str(e)}")
            raise


# Agents for specific lexical elements
class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    async def run(self, user_input):
        logging.info(f"Agent {self.name} processing input: {user_input}")
        try:
            context, task = user_input.split("Context:", 1)
            messages = [
                {"role": "system",
                 "content": f"{self.system_prompt}\n\nUse the following context to inform your response:\n{context}"},
                {"role": "user", "content": task.strip()}
            ]
            response = await openai_post_request(messages, "gpt-4", 500, 0.7)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error in {self.name}: {e}")
            return f"Error: {str(e)}"

class DefinitionAgent(Agent):
    def __init__(self):
        super().__init__("Definition", "Define and explain concepts in the given text.")

# Define agents for different lexical semantics (AssertionAgent, QuestionAgent, etc.)
class AssertionAgent(Agent):
    def __init__(self):
        super().__init__("Assertion", "Process assertions in the given text.")

class QuestionAgent(Agent):
    def __init__(self):
        super().__init__("Question", "Process questions in the given text.")

class HypothesisAgent(Agent):
    def __init__(self):
        super().__init__("Hypothesis", "Process hypotheses in the given text.")

class ThesisAgent(Agent):
    def __init__(self):
        super().__init__("Thesis", "Process thesis statements in the given text.")

class ConditionAgent(Agent):
    def __init__(self):
        super().__init__("Condition", "Process conditional statements in the given text.")

class ConjunctionAgent(Agent):
    def __init__(self):
        super().__init__("Conjunction", "Process conjunctions in the given text.")

class SplitterAgent(Agent):
    def __init__(self):
        super().__init__("Splitter", "Split complex statements into simpler components.")

class ExplorationAgent(Agent):
    def __init__(self):
        super().__init__("Exploration", "Explore concepts mentioned in the given text.")

class ComparisonAgent(Agent):
    def __init__(self):
        super().__init__("Comparison", "Compare the elements specified in the task using the provided context. If the context contains relevant information, use it for the comparison. If not, compare based on your general knowledge.")

class AnalysisAgent(Agent):
    def __init__(self):
        super().__init__("Analysis", "Analyze the given task using the provided context. If the context is relevant, use it to inform your analysis. If not, perform the analysis based on your general knowledge.")

class AgentGraph:
    """Graph structure to manage agents and their interactions"""

    def __init__(self):
        self.agents = {}  # Dictionary to hold agents
        self.graph = nx.DiGraph()  # The graph object to hold dynamic interactions
        self.edge_labels = {}  # Dictionary to hold edge labels (prompts)

    def add_agent(self, agent_name, agent):
        """Add an agent to the graph"""
        self.agents[agent_name] = agent

    async def run(self, input_prompt):
        sub_tasks = await self.decompose_task(input_prompt)

        context = f"Initial question: {input_prompt}\n\n"
        previous_task = "START"
        self.graph.add_node(previous_task, label="Input")
        self.graph.add_node(input_prompt, label="User Prompt")
        self.graph.add_edge(previous_task, input_prompt, label="Initial Input")
        previous_task = input_prompt

        for task in sub_tasks:
            agent_type = self.identify_agent_type(task)
            if agent_type in self.agents:
                logging.info(f"Running task '{task}' with agent: {agent_type}")
                result = await self.agents[agent_type].run(f"Task: {task}\n\nContext:\n{context}")
                context += f"Task: {task}\nResult: {result}\n\n"

                if agent_type not in self.graph:
                    self.graph.add_node(agent_type, label=f"{agent_type.capitalize()} Agent")
                self.graph.add_edge(previous_task, agent_type, label=task)
                previous_task = agent_type

        result_node = "RESULT"
        self.graph.add_node(result_node, label="Final Output")
        self.graph.add_edge(previous_task, result_node, label="Final Processing")

        return context

    async def decompose_task(self, input_prompt):
        """Decompose the main task into subtasks"""
        system_prompt = "You are a task decomposition AI. Given a main task, break it down into 3-5 subtasks."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Decompose this task: {input_prompt}"}
        ]
        response = await openai_post_request(messages, "gpt-3.5-turbo", 150, 0.7)
        subtasks = response['choices'][0]['message']['content'].split('\n')
        return [task.strip() for task in subtasks if task.strip()]

    def identify_agent_type(self, task):
        """Identify the type of agent required based on the task"""
        task_lower = task.lower()

        if "define" in task_lower or "what is" in task_lower:
            return "definition"
        elif "identify" in task_lower or "list" in task_lower:
            return "exploration"
        elif "compare" in task_lower or "versus" in task_lower or "vs" in task_lower:
            return "comparison"
        elif "benefit" in task_lower or "advantage" in task_lower:
            return "analysis"
        elif "if" in task_lower or "when" in task_lower:
            return "condition"
        elif "question" in task_lower or "how" in task_lower or "why" in task_lower:
            return "question"
        else:
            return "analysis"  # Default to analysis for general tasks

    def get_graph_data(self):
        """Returns the graph data in a format suitable for Cytoscape"""
        nodes = [
            {
                'data': {'id': str(node), 'label': self.graph.nodes[node].get('label', str(node))}
            }
            for node in self.graph.nodes()
        ]
        edges = [
            {
                'data': {
                    'source': str(source),
                    'target': str(target),
                    'label': self.graph.edges[source, target].get('label', '')
                }
            }
            for source, target in self.graph.edges()
        ]
        return nodes + edges


# Example usage
async def process_user_prompt(prompt):
    """Process a user input by running it through the agent graph"""
    # Initialize the graph
    agent_graph = AgentGraph()

    # Add existing agents
    agent_graph.add_agent("assertion", AssertionAgent())
    agent_graph.add_agent("question", QuestionAgent())
    agent_graph.add_agent("hypothesis", HypothesisAgent())
    agent_graph.add_agent("thesis", ThesisAgent())
    agent_graph.add_agent("condition", ConditionAgent())
    agent_graph.add_agent("conjunction", ConjunctionAgent())
    agent_graph.add_agent("splitter", SplitterAgent())
    agent_graph.add_agent("definition", DefinitionAgent())
    agent_graph.add_agent("exploration", ExplorationAgent())
    agent_graph.add_agent("comparison", ComparisonAgent())
    agent_graph.add_agent("analysis", AnalysisAgent())

    # Run the user prompt through the graph
    final_output = await agent_graph.run(prompt)

    return agent_graph.get_graph_data()


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        html.H1("Agent Interaction Graph"),
        dbc.Input(id="prompt-input", type="text", placeholder="Enter your prompt here..."),
        dbc.Button("Process", id="process-button", color="primary", className="mt-2"),
        dcc.Loading(
            id="loading",
            type="default",
            children=[
                Cytoscape(
                    id='cytoscape-graph',
                    layout={'name': 'breadthfirst'},
                    style={'width': '100%', 'height': '600px'},
                    elements=[]
                )
            ]
        )
    ])
])


@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('process-button', 'n_clicks'),
    Input('prompt-input', 'value')
)
def update_graph(n_clicks, value):
    if n_clicks is None or not value:
        return []

    # Run the processing asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_user_prompt(value))
    loop.close()

    return result


if __name__ == '__main__':
    app.run_server(debug=True)