import requests
import json
import csv
from datetime import datetime
import os
import random
from multiprocessing import Pool

folder = "scrape/pictures3"
def hex_encode_payload(data):
    # Convert to JSON string and encode to hex
    json_str = json.dumps(data)
    hex_str = json_str.encode('utf-8').hex()
    return hex_str

def search_psa_cards(search_term, page_number=1, page_size=500):
    payload = {
        "filterCategoryId": 1,
        "pricesOnly": True,
        "search": search_term,
        "pageSize": page_size,
        "pageNumber": page_number
    }
    
    hex_encoded = hex_encode_payload(payload)
    url = f"https://www.psacard.com/api/psa/trpc/auctionPrices.search?input=%22{hex_encoded}%22"
    print(f"Searching cards with query: {search_term}, page {page_number}")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch search data: {response.status_code}")

    # take a random sample of 100
    results = response.json()['result']['data']['json']['search']['results']
    print(f"Found {len(results)} results for {search_term}")
    random.shuffle(results)
    results = results[:100]
    return results

def fetch_psa_data(spec_id, page_number=1, page_size=10):
    payload = {
        "specId": spec_id,
        "page": page_number,
        "pageSize": page_size
    }
    
    hex_encoded = hex_encode_payload(payload)

    cookies = {
    'ab.storage.deviceId.f9c2b69f-2136-44e0-a55a-dff72d99aa19': 'g%3AYhSvopkH71QSAVAf3yMJ7jO6dg02%7Ce%3Aundefined%7Cc%3A1740250645739%7Cl%3A1740250645739',
    'ab.storage.sessionId.f9c2b69f-2136-44e0-a55a-dff72d99aa19': 'g%3A73258e85-aa3b-fcab-893e-74a0bef1273b%7Ce%3A1740252445745%7Cc%3A1740250645745%7Cl%3A1740250645745',
    'env': 'prod',
    '_gcl_au': '1.1.1167949792.1740274489',
    '_uutmz': 'source=(direct)|campaign=(direct)|medium=(none)',
    '_uutmzh': '0.4~2025-02-22T20%3A34%3A48.816-05~2025-02-22T20%3A34%3A48.816-05~%3Bnone%3Adirect%3B',
    '_ga': 'GA1.1.281753243.1740274489',
    '__cf_bm': 'xinpV1.2BLqRR5heXkz2ENWVAzh2rnsN8uu6luulZm0-1740276102-1.0.1.1-Ka6NzUBxu4sOoYjiW3fv6fUB2Ia4iO9K82Pu65.IMYC00LMCk1Hxwzx7rJ0noWW4nk9juaE.AWcJ6PC24l0WBGayKarINyTXH6xpsvTp1xE',
    'cf_clearance': 'SlOgvkn0UpUlslN4iCPiHW.ajHE_d4djS6zUt6ad_AE-1740276102-1.2.1.1-gJoMKIcAnhpB4nn5oDs01mwXoB4mt0987PGOhraJhDyq36gzLnm5fMRglEYUv5MZQ5y5SeiIFKYVbQN_wy0msqhWvbaLQpupoGFBg7vaHjiR016l_bawTi4GnbiAlfYB15F7Cz4GVucPOnFc_7T85JuJoS6KXYrWe_wIxn7W2QuahdJfbmqQ3uBRyZdBC7ms_QFnBGOWay5PN_.PmAKr7jlkSS8EyK4eJlP4rFXXCaz43.B0HiWs05ykiyZ1XzCiEtLt4LmcwPQ1frWjIFjnBVdXLvwwNQo9B6f3gtFcRPI',
    '_ga_SZYQ5VYC38': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_FDMLNH35Q0': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_QH8EJ95BRK': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_GGS8NWPYE2': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_8HY0K43CRX': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_LBYZ0MN5NP': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    '_ga_1QVXQ1V575': 'GS1.1.1740274489.1.1.1740276106.0.0.803765999',
    '_ga_5LHM1BS8D8': 'GS1.1.1740274489.1.1.1740276106.0.0.0',
    }

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.psacard.com/auctionprices/search?q=test',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'content-type': 'application/json',
        'locale': 'en',
        'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        # 'Cookie': 'ab.storage.deviceId.f9c2b69f-2136-44e0-a55a-dff72d99aa19=g%3AYhSvopkH71QSAVAf3yMJ7jO6dg02%7Ce%3Aundefined%7Cc%3A1740250645739%7Cl%3A1740250645739; ab.storage.sessionId.f9c2b69f-2136-44e0-a55a-dff72d99aa19=g%3A73258e85-aa3b-fcab-893e-74a0bef1273b%7Ce%3A1740252445745%7Cc%3A1740250645745%7Cl%3A1740250645745; env=prod; _gcl_au=1.1.1167949792.1740274489; _uutmz=source=(direct)|campaign=(direct)|medium=(none); _uutmzh=0.4~2025-02-22T20%3A34%3A48.816-05~2025-02-22T20%3A34%3A48.816-05~%3Bnone%3Adirect%3B; _ga=GA1.1.281753243.1740274489; __cf_bm=xinpV1.2BLqRR5heXkz2ENWVAzh2rnsN8uu6luulZm0-1740276102-1.0.1.1-Ka6NzUBxu4sOoYjiW3fv6fUB2Ia4iO9K82Pu65.IMYC00LMCk1Hxwzx7rJ0noWW4nk9juaE.AWcJ6PC24l0WBGayKarINyTXH6xpsvTp1xE; cf_clearance=SlOgvkn0UpUlslN4iCPiHW.ajHE_d4djS6zUt6ad_AE-1740276102-1.2.1.1-gJoMKIcAnhpB4nn5oDs01mwXoB4mt0987PGOhraJhDyq36gzLnm5fMRglEYUv5MZQ5y5SeiIFKYVbQN_wy0msqhWvbaLQpupoGFBg7vaHjiR016l_bawTi4GnbiAlfYB15F7Cz4GVucPOnFc_7T85JuJoS6KXYrWe_wIxn7W2QuahdJfbmqQ3uBRyZdBC7ms_QFnBGOWay5PN_.PmAKr7jlkSS8EyK4eJlP4rFXXCaz43.B0HiWs05ykiyZ1XzCiEtLt4LmcwPQ1frWjIFjnBVdXLvwwNQo9B6f3gtFcRPI; _ga_SZYQ5VYC38=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_FDMLNH35Q0=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_QH8EJ95BRK=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_GGS8NWPYE2=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_8HY0K43CRX=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_LBYZ0MN5NP=GS1.1.1740274489.1.1.1740276106.0.0.0; _ga_1QVXQ1V575=GS1.1.1740274489.1.1.1740276106.0.0.803765999; _ga_5LHM1BS8D8=GS1.1.1740274489.1.1.1740276106.0.0.0',
    }

    url = f"https://www.psacard.com/api/psa/trpc/auctionPrices.itemDetailAuctionResultsTable?input=%22{hex_encoded}%22"
    print(f"Fetching data for specId: {spec_id}")
    
    response = requests.get(url, headers=headers, cookies=cookies)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    return response.json()

def download_image(image_url, cert_number):
    if not image_url:
        return
        
    # Create pictures directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Extract file extension from URL, default to .jpg if none found
    file_extension = os.path.splitext(image_url)[1] or '.jpg'
    filename = f'{folder}/cert_{cert_number}{file_extension}'
    
    # Skip if file already exists
    if os.path.exists(filename):
        print(f"Image for cert {cert_number} already exists, skipping")
        return
    
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded image for cert {cert_number}")
    except Exception as e:
        print(f"Failed to download image for cert {cert_number}: {str(e)}")

def save_to_csv(data, output_file, spec_info, max_images=10, append=False):
    # Extract the individual sales from the response
    sales = data['result']['data']['json']['spec']['historicalAuctionInfo']['pastListingInfo']['individualSales']
    
    # Define CSV headers
    headers = [
        'specId',
        'setName',
        'collectibleYear',
        'collectibleSubject',
        'auctionItemId',
        'auctionHouse',
        'imageUrl',
        'listingType',
        'dateOfSale',
        'grade',
        'salePrice',
        'certNumber'
    ]
    
    # Write to CSV
    mode = 'a' if append else 'w'
    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not append:  # Only write header if this is a new file
            writer.writeheader()
        for sale in sales:
            writer.writerow({
                'specId': spec_info['specId'],
                'setName': spec_info['setName'],
                'collectibleYear': spec_info['collectibleYear'],
                'collectibleSubject': spec_info['collectibleSubject'],
                'auctionItemId': sale['auctionItemId'],
                'auctionHouse': sale['auctionHouse'],
                'imageUrl': sale['imageUrl'],
                'listingType': sale['listingType'],
                'dateOfSale': sale['dateOfSale'],
                'grade': sale['grade'],
                'salePrice': sale['salePrice'],
                'certNumber': sale['certNumber']
            })
    
    # Prepare image download tasks
    download_tasks = []
    for sale in sales[:max_images]:
        if sale['imageUrl']:
            download_tasks.append((sale['imageUrl'], sale['certNumber']))
    
    if download_tasks:
        # Use process pool to download images in parallel
        with Pool() as pool:
            pool.starmap(download_image, download_tasks)
            print(f"Downloaded {len(download_tasks)} images in parallel")

def main():
    try:
        # Generate single output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"psa_sales.csv"
        
        specs = []
        # Search for baseball cards
        # Spread out throughout the years, 2000-2020
        for i in range(1, 30):
            search_results = search_psa_cards(page_number=1, search_term=f"{1980 + i}")
            specs.extend(search_results)
            # scramble the mix
            random.shuffle(specs)
        print(f"Found {len(specs)} total specs")
        
        # Process each spec
        for i, spec in enumerate(specs):
            spec_id = spec['specId']
            print(f"\nProcessing spec {spec_id}: {spec['collectibleDescription']} ({i+1}/{len(specs)})")
            
            try:
                page_size = 10
                # Fetch the data for this spec
                data = fetch_psa_data(spec_id, page_number=1, page_size=page_size)
                
                # Save to CSV and download up to 10 images
                save_to_csv(data, output_file, spec_info=spec, max_images=page_size, append=(i > 0))
                print(f"Data successfully appended to {output_file}")
                
            except Exception as e:
                print(f"Error processing spec {spec_id}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
