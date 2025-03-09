from bs4 import BeautifulSoup

# Provide the path to your local HTML file
file_path = "your file path here"

# Open the file and read its content
with open(file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, "html.parser")

# Remove the content within div elements with specific class names
for div in soup.select('div#skip-to-main, div.row, div.utility, div.main, div.mobile, div.links, div.secondary, div.bottom'):
    div.decompose()

# Extract the text content while removing tags, scripts, and styles
text_only = soup.get_text(strip=True, separator="\n")

print(text_only)
