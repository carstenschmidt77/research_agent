import arxiv
from Bio import Entrez
from typing import List, Dict, Any

def search_arxiv(*, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for papers in arXiv"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in client.results(search):
            authors = [a.name for a in result.authors]
            results.append({
                "title": result.title,
                "authors": authors[:3],
                "year": result.published.year,
                "url": result.pdf_url
            })
        
        return results
    except Exception as e:
        print(f"arXiv Error: {str(e)}")
        return []

def search_pubmed(*, query: str, api_key: str = None, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for papers in PubMed/PMC using the Entrez API"""
    try:
        Entrez.email = "your_email@example.com"
        if api_key:
            Entrez.api_key = api_key
        
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            datetype="pdat",
            reldate=365
        )
        results = Entrez.read(handle)
        handle.close()
        
        ids = results["IdList"]
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        papers = Entrez.read(handle)["PubmedArticle"]
        
        output = []
        for paper in papers:
            article = paper["MedlineCitation"]["Article"]
            title = article.get("ArticleTitle", "No title available")
            authors = [f"{a.get('LastName', '')} {a.get('Initials', '')}" 
                      for a in article.get("AuthorList", [])]
            pub_date = article["Journal"]["JournalIssue"]["PubDate"].get("Year", "Unknown date")
            pmid = paper["MedlineCitation"].get("PMID", "")
            
            output.append({
                "title": title,
                "authors": authors[:3],
                "year": pub_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
            
        return output
    
    except Exception as e:
        print(f"PubMed Error: {str(e)}")
        return [] 