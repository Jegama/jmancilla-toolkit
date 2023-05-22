import requests, re

def find_urls_in_webpage(url):
    # Fetch the web page content
    response = requests.get(url)
    webpage_content = response.text

    # Define the regular expression pattern
    pattern = r'https:\/\/support\.roku\.com\/article\/\d+'

    # Find all URLs matching the pattern
    urls = re.findall(pattern, webpage_content)

    return urls

urls = find_urls_in_webpage('https://support.roku.com/sitemap.xml')
print(f'\nFound {len(urls)} urls')

# remove duplicates from list
urls = list(dict.fromkeys(urls))
print(f'\nFound {len(urls)} unique urls')

for i in urls[:10]:
    print(i)