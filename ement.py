from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import spacy
from ement_llm_memory import EmentMemory  # EMENT API assumption

# Import the new GitHub search function
from search_tools import search_github_repos

# Define a simple decorator named 'tool' if the original one isn't available
def tool(func):
    # Depending on your needs, wrap or transform the function here.
    # For now, it simply returns the function as-is.
    return func

# 1. EMENT Initialization
nlp = spacy.load("en_core_web_lg")
ement = EmentMemory(llm_embeddings="text-embedding-3-small")

# 2. Pydantic Models
class MemoryEntry(BaseModel):
    entities: list[str] = Field(..., description="Liste extrahierter Entitäten")
    context: str = Field(..., description="Vollständiger Kontexttext")
    embedding: list[float] = Field(..., description="Vektor-Embedding")

class EnhancedAgentResponse(BaseModel):
    answer: str
    relevant_memories: list[MemoryEntry]
    confidence: float = Field(..., ge=0, le=1)

# 3. Dependency class for EMENT
@dataclass
class EmentDependencies:
    memory: EmentMemory
    nlp_model: spacy.language.Language

# 4. Tools for EMENT integration
@tool
async def extract_entities(ctx: RunContext[EmentDependencies], text: str) -> list[str]:
    """Extrahiert Entitäten aus Text und speichert in EMENT"""
    doc = ctx.deps.nlp_model(text)
    entities = [ent.text for ent in doc.ents]
    embedding = await ctx.deps.memory.generate_embedding(text)
    ctx.deps.memory.store(
        context=text,
        entities=entities,
        embedding=embedding
    )
    return entities

@tool
async def query_memory(ctx: RunContext[EmentDependencies], query: str) -> list[MemoryEntry]:
    """Durchsucht EMENT nach relevanten Erinnerungen"""
    results = ctx.deps.memory.search(
        query=query,
        entity_threshold=0.7,
        embedding_threshold=0.85
    )
    return [
        MemoryEntry(
            entities=res["entities"],
            context=res["context"],
            embedding=res["embedding"]
        ) for res in results
    ]

# New GitHub search tool
@tool
async def search_github_repos_tool(
    ctx: RunContext[EmentDependencies],
    query: str,
    max_results: int = 5,
    token: str = None
) -> list[dict]:
    """
    Wrapper around the search_github_repos function.
    Execute a GitHub repository search for the given 'query'.
    """
    return search_github_repos(query=query, max_results=max_results, token=token)

# 5. Agent configuration
agent = Agent(
    "openai:gpt-4o",
    deps_type=EmentDependencies,
    result_type=EnhancedAgentResponse,
    system_prompt=(
        "Nutze EMENT, um Kontext aus früheren Interaktionen abzurufen. "
        "Analysiere Entitäten und füge relevante Erinnerungen in Antworten ein."
    ),
    tools=[extract_entities, query_memory, search_github_repos_tool]
)

# 6. Dynamic system prompt with memory
@agent.system_prompt
async def memory_context(ctx: RunContext[EmentDependencies]) -> str:
    memories = await query_memory(ctx, ctx.user_prompt)
    context = "\n".join([m.context for m in memories[:3]])
    return f"Relevanter Kontext:\n{context}\nAntworte präzise unter Berücksichtigung dieser Informationen"

# 7. Example run
async def main():
    deps = EmentDependencies(
        memory=ement,
        nlp_model=nlp
    )
    
    # First interaction (store info)
    result = await agent.run(
        "Meine Katze Mimi hat heute Geburtstag. Sie ist 3 Jahre alt geworden.",
        deps=deps
    )
    
    # Second query (retrieve from memory)
    result = await agent.run(
        "Wie alt ist meine Katze?",
        deps=deps
    )
    
    # Third usage example: search GitHub repos
    result = await agent.run(
        "Suche GitHub Repositories zu Transformers in Python",
        deps=deps
    )
    
    print(f"Antwort: {result.data.answer}")
    print(f"Verwandte Erinnerungen: {len(result.data.relevant_memories)}")
    print(f"Konfidenz: {result.data.confidence:.2%}")

# Erwartete Ausgabe:
# Antwort: Ihre Katze Mimi ist 3 Jahre alt.
# Verwandte Erinnerungen: 1
# Konfidenz: 95.00%