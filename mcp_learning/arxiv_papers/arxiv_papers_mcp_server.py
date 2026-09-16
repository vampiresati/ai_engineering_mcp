import arxiv
import json
import os
from dotenv import load_dotenv
from fastmcp import FastMCP


load_dotenv()

from fastmcp.server.auth.providers.keycloak import KeycloakAuthProvider
auth = KeycloakAuthProvider(
    realm_url="http://localhost:8080/realms/mcp-realm",
    base_url="http://localhost:8000",
)

mcp = FastMCP("Research", auth=auth)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(BASE_DIR, "papers")


# ============================================================
# SAVE PAPERS
# ============================================================

def create_directory_about_the_paper(PAPER_DIR, topic, papers):

    path = os.path.join(
        PAPER_DIR,
        topic.lower().replace(" ", "_")
    )

    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(
        path,
        "papers_info.json"
    )

    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)

    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []

    for paper in papers:

        paper_id = paper.get_short_id()

        paper_ids.append(paper_id)

        paper_info = {
            "title": paper.title,
            "authors": [
                author.name
                for author in paper.authors
            ],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "paper_id": paper_id,
            "published": str(paper.published.date())
        }

        papers_info[paper_id] = paper_info

    # Save once after processing all papers
    with open(file_path, "w") as json_file:
        json.dump(
            papers_info,
            json_file,
            indent=2
        )

    print(
        f"Results are saved in: {file_path}"
    )

    return paper_ids


# ============================================================
# TOOL 1 - EXTRACT PAPER
# ============================================================

@mcp.tool
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper
    across all topic directories.

    Args:
        paper_id: The ID of the paper to look for.

    Returns:
        JSON string with paper information if found.
    """

    if not os.path.exists(PAPER_DIR):
        return (
            f"There's no saved information "
            f"related to paper {paper_id}."
        )

    for item in os.listdir(PAPER_DIR):

        item_path = os.path.join(
            PAPER_DIR,
            item
        )

        if os.path.isdir(item_path):

            file_path = os.path.join(
                item_path,
                "papers_info.json"
            )

            if os.path.isfile(file_path):

                try:

                    with open(
                        file_path,
                        "r"
                    ) as json_file:

                        papers_info = json.load(
                            json_file
                        )

                    if paper_id in papers_info:

                        return json.dumps(
                            papers_info[paper_id],
                            indent=2
                        )

                except (
                    FileNotFoundError,
                    json.JSONDecodeError
                ) as e:

                    print(
                        f"Error reading {file_path}: {str(e)}"
                    )

                    continue

    return (
        f"There's no saved information "
        f"related to paper {paper_id}."
    )


# ============================================================
# TOOL 2 - SEARCH PAPERS
# ============================================================

@mcp.tool
def search_papers(
    topic: str,
    max_results: int = 5
):

    print(f"Searching arXiv: {topic}")
    print(f"Maximum results: {max_results}")

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=1
    )

    papers = client.results(search)

    try:

        paper_ids = create_directory_about_the_paper(
            PAPER_DIR,
            topic,
            papers
        )

        return {
            "success": True,
            "paper_ids": paper_ids
        }

    except arxiv.HTTPError as e:

        return {
            "success": False,
            "error": "arXiv API rate limit (HTTP 429)",
            "message": str(e)
        }


# ============================================================
# RESOURCE 1
# papers://folders
# ============================================================

@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
    List all available topic folders in the papers directory.
    """

    folders = []

    if os.path.exists(PAPER_DIR):

        for topic_dir in os.listdir(PAPER_DIR):

            topic_path = os.path.join(
                PAPER_DIR,
                topic_dir
            )

            if os.path.isdir(topic_path):

                papers_file = os.path.join(
                    topic_path,
                    "papers_info.json"
                )

                if os.path.exists(papers_file):

                    folders.append(topic_dir)

    content = "# Available Topics\n\n"

    if folders:

        for folder in sorted(folders):

            content += f"- {folder}\n"

        content += (
            "\nUse the topic resource "
            "`papers://{topic}` to access papers "
            "in a topic.\n"
        )

    else:

        content += "No topics found.\n"

    return content


# ============================================================
# RESOURCE 2
# papers://{topic}
# ============================================================

@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    Get detailed information about papers
    on a specific topic.

    Args:
        topic: The research topic.
    """

    topic_dir = (
        topic
        .lower()
        .replace(" ", "_")
    )

    papers_file = os.path.join(
        PAPER_DIR,
        topic_dir,
        "papers_info.json"
    )

    if not os.path.exists(papers_file):

        return (
            f"# No papers found for topic: {topic}\n\n"
            "Try searching for papers on this topic first."
        )

    try:

        with open(
            papers_file,
            "r"
        ) as f:

            papers_data = json.load(f)

        content = (
            f"# Papers on "
            f"{topic.replace('_', ' ').title()}\n\n"
        )

        content += (
            f"Total papers: "
            f"{len(papers_data)}\n\n"
        )

        for paper_id, paper_info in papers_data.items():

            content += (
                f"## {paper_info['title']}\n"
            )

            content += (
                f"- **Paper ID:** "
                f"{paper_id}\n"
            )

            content += (
                f"- **Authors:** "
                f"{', '.join(paper_info['authors'])}\n"
            )

            content += (
                f"- **Published:** "
                f"{paper_info['published']}\n"
            )

            content += (
                f"- **PDF URL:** "
                f"{paper_info['pdf_url']}\n\n"
            )

            summary = paper_info.get(
                "summary",
                "No summary available."
            )

            content += (
                f"### Summary\n"
                f"{summary[:500]}...\n\n"
            )

            content += "---\n\n"

        return content

    except json.JSONDecodeError:

        return (
            f"# Error reading papers data for {topic}\n\n"
            "The papers data file is corrupted."
        )


# ============================================================
# RESOURCE 3
# papers://{topic}/{paper_id}
# ============================================================

@mcp.resource("papers://{topic}/{paper_id}")
def get_paper(topic: str,paper_id: str) -> str:
    """
    Get detailed information about a specific paper.
    Args:
        topic: Topic containing the paper.
        paper_id: arXiv paper ID.
    """
    topic_dir = (topic.lower().replace(" ", "_"))
    papers_file = os.path.join(PAPER_DIR,topic_dir,"papers_info.json")
    if not os.path.exists(papers_file):
        return (
            f"# Topic Not Found\n\n"
            f"No papers found for topic: {topic}"
        )

    try:
        with open(papers_file,"r") as f:
            papers_data = json.load(f)
    except json.JSONDecodeError:
        return (
            f"# Error\n\n"
            f"The papers data file for "
            f"'{topic}' is corrupted."
        )

    paper_info = papers_data.get(paper_id)

    if not paper_info:

        return (
            f"# Paper Not Found\n\n"
            f"No paper with ID `{paper_id}` "
            f"was found in topic `{topic}`."
        )

    content = (
        f"# {paper_info['title']}\n\n"
    )

    content += (
        f"- **Paper ID:** "
        f"{paper_id}\n"
    )

    content += (
        f"- **Authors:** "
        f"{', '.join(paper_info.get('authors', []))}\n"
    )

    content += (
        f"- **Published:** "
        f"{paper_info.get('published', 'Unknown')}\n"
    )

    content += (
        f"- **PDF URL:** "
        f"{paper_info.get('pdf_url', 'Not available')}\n\n"
    )

    content += "## Summary\n\n"

    content += (
        paper_info.get(
            "summary",
            "No summary available."
        )
    )

    content += "\n"

    return content


# ============================================================
# PROMPT
# ============================================================

@mcp.prompt()
def generate_search_prompt(
    topic: str,
    num_papers: int = 5
) -> str:
    """
    Generate a research prompt for finding and
    discussing academic papers.
    """

    return f"""
Search for {num_papers} academic papers about
'{topic}' using the search_papers tool.
Follow these instructions:
1. First search for papers:
   search_papers(
       topic="{topic}",
       max_results={num_papers}
   )
2. For each paper found, use extract_info to
   retrieve its saved information.
3. Organize the following information for each paper:
   - Paper title
   - Authors
   - Publication date
   - Brief summary of key findings
   - Main contributions or innovations
   - Methodologies used
   - Relevance to the topic "{topic}"
4. Then provide a research-level synthesis containing:
   - Overview of the current research on "{topic}"
   - Common themes
   - Common methodologies
   - Important trends
   - Research gaps
   - Future research directions
5. Organize the answer using:
   ## Research Overview
   ## Papers
   ### Paper 1
   - Title
   - Authors
   - Publication date
   - Summary
   - Contributions
   - Methodology
   - Relevance
   ### Paper 2
   ...
   ## Common Themes
   ## Research Trends
   ## Research Gaps
   ## Future Directions

Do not invent information that is not available
from the papers.
"""
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )

    # For HTTP:
    #
    # mcp.run(
    #     transport="http",
    #     host="127.0.0.1",
    #     port=8001
    # )
