import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from search_tools import search_arxiv, search_pubmed

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

agent = Agent(
    model,
    system_prompt='''You are a helpful research assistant. You can search for academic papers on arXiv and PubMed.
    Be concise in your responses and always include relevant paper titles and authors when answering research questions.''',
    tools=[search_arxiv, search_pubmed]
)

def main():
    theme = input("Enter your research topic: ")
    print(f"\nSearching for papers about: '{theme}'...\n")
    result = agent.run_sync(f'Find recent papers about {theme} and their applications. Use the tools pubmed only for medical papers. Please state the tool you used in your response.')
    print(result.data)

if __name__ == "__main__":
    main()
