import pandas as pd
import csv
#import beautifulsoup4 as bs

def get_nifty50_data():
    # Step 1: Scrape the Nifty 50 table from Wikipedia
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    
    # We use match="Symbol" to specifically target the table containing the ticker symbols
    tables = pd.read_html(url, match="Symbol", storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
    df = tables[0]
    
    # Step 2: Extract the symbols and format them for Yahoo Finance
    # Indian stocks on Yahoo Finance require the '.NS' (National Stock Exchange) suffix
    raw_symbols = df['Symbol'].tolist()
    nifty50_tickers = [symbol + ".NS" for symbol in raw_symbols]
    
    print(f"Successfully extracted {len(nifty50_tickers)} tickers.")
    print(f"Sample tickers: {nifty50_tickers[:5]}\n")

    return nifty50_tickers
    

    
'''
# Run the function
nifty_data = get_nifty50_data()

with open('nifty50_tickers.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ticker'])
    for ticker in nifty_data:
        writer.writerow([ticker])

'''



import requests
import io

def get_nifty_smallcap50_data():
    # Step 1: Fetch the official Nifty Smallcap 50 CSV from the NSE archives
    url = "https://archives.nseindia.com/content/indices/ind_niftysmallcap50list.csv"
    
    # We use headers to mimic a browser, as the NSE website blocks automated scraper bots
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Step 2: Read the CSV content into a pandas DataFrame
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Step 3: Extract the symbols and format them for Yahoo Finance
        raw_symbols = df['Symbol'].tolist()
        smallcap50_tickers = [symbol + ".NS" for symbol in raw_symbols]
        
        print(f"Successfully extracted {len(smallcap50_tickers)} tickers.")
        print(f"Sample tickers: {smallcap50_tickers[:5]}\n")
        
        return smallcap50_tickers
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
        return []

'''

# Run the function
smallcap_data = get_nifty_smallcap50_data()

# Save the formatted tickers to a CSV
if smallcap_data:
    with open('nifty_smallcap50_tickers.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ticker'])
        for ticker in smallcap_data:
            writer.writerow([ticker])
    print("Data successfully saved to 'nifty_smallcap50_tickers.csv'")



'''


def get_nifty_midcap50_data():
    # Step 1: Fetch the official Nifty Midcap 50 CSV from the NSE archives
    url = "https://archives.nseindia.com/content/indices/ind_niftymidcap50list.csv"
    
    # We use headers to mimic a browser, as the NSE website blocks automated scraper bots
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Step 2: Read the CSV content into a pandas DataFrame
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Step 3: Extract the symbols and format them for Yahoo Finance
        raw_symbols = df['Symbol'].tolist()
        midcap50_tickers = [symbol + ".NS" for symbol in raw_symbols]
        
        print(f"Successfully extracted {len(midcap50_tickers)} tickers.")
        print(f"Sample tickers: {midcap50_tickers[:5]}\n")
        
        return midcap50_tickers
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
        return []

'''

# Run the function
midcap_data = get_nifty_midcap50_data()

# Save the formatted tickers to a CSV
if midcap_data:
    with open('nifty_midcap50_tickers.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ticker'])
        for ticker in midcap_data:
            writer.writerow([ticker])
    print("Data successfully saved to 'nifty_midcap50_tickers.csv'")


    '''



def get_list_from_csv_pandas(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Convert the 'Ticker' column directly to a Python list
    ticker_list = df['Ticker'].tolist()
    
    return ticker_list

# Run it
niftysmlcap = get_list_from_csv_pandas('./Tickers/nifty_smallcap50_tickers.csv')
niftymidcap = get_list_from_csv_pandas('./Tickers/nifty_midcap50_tickers.csv')
nifty50 = get_list_from_csv_pandas('./Tickers/nifty50_tickers.csv')
print(niftysmlcap)
print(niftymidcap)
print(nifty50)