# -*- coding: utf-8 -*-
import json
import requests
from time import sleep
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver

# Sahand Sabour - 山姆
ID = 2022380024


def screenshot(driver, SE, QN, Extra=False):
    w = driver.execute_script("return document.body.parentNode.scrollWidth")
    h = driver.execute_script("return document.body.parentNode.scrollHeight")
    driver.set_window_size(w, h)
    driver.get_screenshot_as_file(f"SE_{SE}_{QN}_{ID}{'_1' if Extra else ''}.png")


def crawl(driver, url):
    driver.get(url)
    sleep(2)


def extract_search_res(driver, SE, QN, url):
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    if SE == "ecosia":
        search_res = soup.find_all("div", {"class": "result__body"})
        if len(search_res) < 10:
            crawl(driver, url + "&p=1")
            screenshot(driver, SE, QN, True)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            search_res += soup.find_all("div", {"class": "result__body"})
    else:
        search_res = soup.find_all("li", {"class": "b_algo"})
    output = []

    for res in search_res:
        output.append(
            {
                "rank": len(output) + 1,
                "title": res.find("h3").text if SE == "google" else res.find("h2").text,
                "url": res.find("a")["href"],
            }
        )
        if len(output) == 10:
            with open(f"SE_{SE}_{QN}_{ID}.json", "w", encoding="utf8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            break

    return output


def extract_url(res, SE, QN):
    for r in tqdm(res):
        url = r["url"]
        out = requests.get(url)
        with open(
            f"html/SE_{SE}_{QN}_{r['rank']}_{ID}.html", "w", encoding="utf8"
        ) as f:
            f.write(out.text)


if __name__ == "__main__":
    driver = webdriver.Safari()
    driver.maximize_window()
    search_engines = ["bing"]
    domains = ["com"]
    queries = json.load(open(f"QD_{ID}.json", "r"))
    for SE, domain in zip(search_engines, domains):
        for q in queries:
            QN = q["queryNum"]
            url = f"https://www.{SE}.{domain}/search?q={q['query']}&count=15&ensearch=1&hl=en"
            crawl(driver, url)
            screenshot(driver, SE, QN)
            res = extract_search_res(driver, SE, q["queryNum"], url)
            extract_url(res, SE, QN)
    driver.quit()
