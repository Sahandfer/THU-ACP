# Web Information Retreival - Assignment 1
In this assingment, we were tasked to create a crawler that crawls the content of the top-10 search results from 2 arbitrary search engines.

### Overview of the Program

The program uses the Python **Selenium** library to leverage a browser simulator. For each search engine, calls for searches of the designated query are called within the simulator and the corresponding screenshot is taken using the Selenium API. Accordingly, we use the **BeautifulSoup** library to parse the results landing page and extract the top-10 search results. Note that if there are less than 10 results shown on the first page, we would go to the second page. Accordingly, we would take an additional screenshot for the second page that has the suffix "_1".  Then, we loop through the search results, leverage the **Requests** library to send a simple GET request and get the content of each webpage.

## Guide
You can use the following command to run the code:
```shell
python Crawler.py
```

The program needs the query file `QD_2022380024.json` to run.