from playwright.sync_api import sync_playwright

def extract_text_from_div(url, class_name):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        page.goto(url)
        page.wait_for_load_state('networkidle')

        div_element = page.query_selector(f".{class_name}")
        if div_element:
            text_content = div_element.inner_text()
        else:
            print(f"No element found with class: {class_name}")

        browser.close()
    return text_content

extract_text_from_div('https://support.roku.com/article/208756478', 'article-content-wrapper')