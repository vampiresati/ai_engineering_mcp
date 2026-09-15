import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv
PAPER_DIR = "papers"

def print_detail_about_paper(paper):
    print('_____')
    print(paper.title,paper.published)
    print(paper.summary)
    print(paper.pdf_url)
    print(paper.entry_id)
    print('*****')

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    Args:
        paper_id: The ID of the paper to look for
    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."

def create_directory_about_the_paper(PAPER_DIR,topic,papers):
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_id=paper.get_short_id()
        paper_ids.append(paper_id)
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'paper_id':paper_id,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

        with open(file_path, "w") as json_file:
            json.dump(papers_info, json_file, indent=2)
        print(f"Results are saved in: {file_path}")
    return paper_ids
def search_papers(topic="mcp",max_results=5):
    """
    Search for papers on arXiv based on a topic and store their information.
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
    Returns:
        List of paper IDs found in the search
    """
    client = arxiv.Client()
    search = arxiv.Search(query = topic,max_results = max_results,sort_by = arxiv.SortCriterion.SubmittedDate)
    papers = client.results(search)
    #for paper in papers:
    #    print_detail_about_paper(paper)
    paper_ids=create_directory_about_the_paper(PAPER_DIR,topic,papers)
    print(paper_ids)
    return paper_ids
if __name__=="__main__":
#   paper_ids=search_papers("mcp",2)
    print(extract_info("2609.15906v1"))
