import requests
from bs4 import BeautifulSoup

def fetch_et_news(topic="Tata-Motors", limit=5):
    """
    Fetch top news headlines from Economic Times for a given topic.

    Parameters:
    - topic (str): Topic for which to fetch news (use hyphenated string like 'Tata-Motors')
    - limit (int): Number of headlines to return

    Returns:
    - List[Dict]: Each dict contains 'title' and 'link'
    """
    base_url = "https://economictimes.indiatimes.com/topic/"
    url = base_url + topic
    res = requests.get(url)
    
    if res.status_code != 200:
        print(f"❌ Failed to fetch news for topic: {topic}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    headlines = []

    for headline in soup.select(".eachStory h3")[:limit]:
        title = headline.get_text(strip=True)
        parent = headline.find_parent("a")
        link = "https://economictimes.indiatimes.com" + parent["href"] if parent else "N/A"
        headlines.append({"title": title, "link": link})

    return headlines
