from bs4 import BeautifulSoup
from typing import List, Tuple
import json

def extract_h1_tags_with_links(file_path: str) -> List[Tuple[str, str]]:
    """
    Extract the text and associated links from all <h1> tags with a specific class in an HTML file.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the text of the <h1> tag and its associated link.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    h1_tags = soup.find_all("h1", class_="ogp-askgov-question-card-v2-title")

    extracted_data = []
    for h1 in h1_tags:
        text = h1.get_text(strip=True)
        link = None

        if h1.find("a"):
            link = h1.find("a")["href"]
        else:
            parent = h1.find_parent("a")
            if parent and "href" in parent.attrs:
                link = parent["href"]

        extracted_data.append((text, link if link else "No link found"))

    return extracted_data

def save_to_json(data: List[Tuple[str, str]], output_file: str) -> None:
    """
    Save the extracted data to a JSON file.

    Args:
        data (List[Tuple[str, str]]): The data to save.
        output_file (str): The path to the output JSON file.
    """
    json_data = [{"text": text, "link": link} for text, link in data]
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)
    print(f"saved to {len(data)} links to json")

def main() -> None:
    """
    Main function to extract, print, and save text and links from <h1> tags in an HTML file.
    """
    file_path = "data/ICA_FAQ.html"
    output_file = "data/ICA_FAQ.json"
    h1_data = extract_h1_tags_with_links(file_path)

    for text, link in h1_data:
        print(f"Text: {text}")
        print(f"Link: {link}")
        print("-" * 50)

    save_to_json(h1_data, output_file)

if __name__ == "__main__":
    main()