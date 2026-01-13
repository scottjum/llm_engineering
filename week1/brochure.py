`"""
Brochure Generator - Creates company brochures from website content using LLMs.

This script scrapes a company's website, identifies relevant pages, and generates
a marketing brochure using an LLM.

Usage:
    uv run brochure.py <company_name> <url>
    uv run brochure.py "HuggingFace" "https://huggingface.co"
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from scraper import fetch_website_contents, fetch_website_links

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client
MODEL = os.getenv("BROCHURE_MODEL", "gpt-oss:20b")
openai_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key=os.getenv("OLLAMA_API_KEY"),
)

# System prompt for selecting relevant links
LINK_SYSTEM_PROMPT = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

# System prompt for generating the brochure
#BROCHURE_SYSTEM_PROMPT = """
#You are an assistant that analyzes the contents of several relevant pages from a company website
#and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
#Respond in markdown without code blocks.
#Include details of company culture, customers and careers/jobs if you have the information.
#"""
BROCHURE_SYSTEM_PROMPT = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

def get_links_user_prompt(url: str) -> str:
    """Generate the user prompt for link selection."""
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt


def select_relevant_links(url: str, model: str) -> dict:
    """Use LLM to select relevant links from a webpage."""
    print(f"Selecting relevant links for {url} by calling {model}")
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {"role": "user", "content": get_links_user_prompt(url)},
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    print(f"Found {len(links['links'])} relevant links")
    return links


def fetch_page_and_all_relevant_links(url: str, model: str) -> str:
    """Fetch the landing page and all relevant linked pages."""
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url, model=model)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links["links"]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


def get_brochure_user_prompt(company_name: str, url: str, model: str) -> str:
    """Generate the user prompt for brochure creation."""
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.

"""
    user_prompt += fetch_page_and_all_relevant_links(url, model=model)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt


def create_brochure(company_name: str, url: str, model: str) -> str:
    """Create a brochure for the given company (non-streaming)."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, model=model)},
        ],
    )
    return response.choices[0].message.content


def stream_brochure(company_name: str, url: str, model: str) -> str:
    """Create a brochure with streaming output to the terminal."""
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, model=model)},
        ],
        stream=True,
    )
    response = ""
    print("\n" + "=" * 60)
    print("GENERATED BROCHURE")
    print("=" * 60 + "\n")
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        sys.stdout.write(content)
        sys.stdout.flush()
    print("\n\n" + "=" * 60)
    return response


def main():
    """Main entry point for the brochure generator."""
    parser = argparse.ArgumentParser(
        description="Generate a company brochure from website content using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run brochure.py "HuggingFace" "https://huggingface.co"
    uv run brochure.py --no-stream "Anthropic" "https://anthropic.com"
        """,
    )
    parser.add_argument("company_name", help="Name of the company")
    parser.add_argument("url", help="URL of the company's website")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (print all at once)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to use (default: {MODEL})",
    )

    args = parser.parse_args()

    model = args.model

    print(f"\n🏢 Generating brochure for: {args.company_name}")
    print(f"🌐 Website: {args.url}")
    print(f"🤖 Model: {model}\n")

    if args.no_stream:
        brochure = create_brochure(args.company_name, args.url, model=model)
        print("\n" + "=" * 60)
        print("GENERATED BROCHURE")
        print("=" * 60 + "\n")
        print(brochure)
    else:
        stream_brochure(args.company_name, args.url, model=model)


if __name__ == "__main__":
    main()

