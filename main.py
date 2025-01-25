import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from search_tools import search_arxiv, search_pubmed, search_github
from datetime import datetime
import dotenv

# Load environment variables from env.local
dotenv.load_dotenv('env.local')

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

agent = Agent(
    model,
    system_prompt='''You are a helpful research assistant. You can search for academic papers on arXiv and PubMed, and repositories on GitHub.
    Based on the topic, choose the most appropriate search tool:
    - For medical/health topics: use PubMed
    - For technical/scientific papers: use arXiv
    - For software/code projects: use GitHub
    Be concise in your responses and always include relevant titles and authors.''',
    tools=[search_arxiv, search_pubmed, search_github]
)

summary_agent = Agent(
    model,
    system_prompt='''Act as an expert summarizer with advanced NLP capabilities. Follow these steps meticulously:

    Preprocess the Input:
    Remove ads, duplicates, and irrelevant sections.
    Segment [CONTENT] into logical chunks (e.g., paragraphs, sections).
    Extract Core Information:
    Identify key entities, themes, and sentiment using NLP.
    Rank sentences by salience (position, semantic relevance, keyword frequency).
    Choose Summarization Strategy:
    If the content is technical/factual (e.g., research, news), use extractive summarization (direct quotes of key sentences).
    If the content is narrative/creative (e.g., stories, opinions), use abstractive summarization (paraphrase concepts).
    Optimize for Context:
    Adapt to the [DOMAIN] (e.g., medical, legal, tech) for terminology accuracy.
    Resolve ambiguities (e.g., link pronouns like 'it' or 'they' to their entities).
    Structure the Summary:
    Length: [DESIRED LENGTH] (e.g., 10% of original, 3 sentences, or 300 words).
    Format: Start with a 1-sentence TL;DR, then expand key points hierarchically.
    Use connectors like "However," "Additionally," or "In conclusion" for flow.
    Validate & Refine:
    Cross-check critical claims against verified sources (if available).
    Remove redundancies and simplify jargon.
    Preserve the original tone (e.g., formal, casual) and intent.
    Ethical Guardrails:
    Audit for bias (gender, race, etc.) and neutralize if detected.
    Never introduce new claims unsupported by the original content.
    
    Proceed step-by-step and explain your reasoning before delivering the final summary.''',
)

def save_to_markdown(content: str, filename: str):
    """Save content to a markdown file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def main():
    theme = input("Enter your research topic: ")
    print(f"\nSearching for information about: '{theme}'...\n")
    
    # Get search results
    search_result = agent.run_sync(f'Find recent information about {theme}. Choose the most appropriate search tool based on the topic.')
    print("Search completed. Saving results...")
    
    # Save search results
    search_filename = save_to_markdown(search_result.data, "search_results")
    print(f"Search results saved to: {search_filename}")
    
    # Generate and save summary
    print("\nGenerating summary...\n")
    summary_result = summary_agent.run_sync(f'Summarize the following search results: {search_result.data}')
    summary_filename = save_to_markdown(summary_result.data, "summary")
    print(f"Summary saved to: {summary_filename}")

if __name__ == "__main__":
    main()
