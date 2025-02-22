import requests
import json
import csv
from datetime import datetime
import os

def hex_encode_payload(data):
    # Convert to JSON string and encode to hex
    json_str = json.dumps(data)
    hex_str = json_str.encode('utf-8').hex()
    return hex_str

def search_psa_cards(page_number=1, page_size=500, search_term="2020"):
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
    
    return response.json()

def fetch_psa_data(spec_id):
    payload = {
        "specId": spec_id,
        "page": 1,
        "pageSize": 50
    }
    
    hex_encoded = hex_encode_payload(payload)
    url = f"https://www.psacard.com/api/psa/trpc/auctionPrices.itemDetailAuctionResultsTable?input=%22{hex_encoded}%22"
    print(f"Fetching data for specId: {spec_id}")
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    
    return response.json()

def download_image(image_url, cert_number):
    if not image_url:
        return
        
    # Create pictures directory if it doesn't exist
    os.makedirs('pictures', exist_ok=True)
    
    # Extract file extension from URL, default to .jpg if none found
    file_extension = os.path.splitext(image_url)[1] or '.jpg'
    filename = f'pictures/cert_{cert_number}{file_extension}'
    
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

def save_to_csv(data, output_file, spec_info, max_images=50, append=False):
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
    
    image_count = 0
    
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
            
            # Download the image for this sale (up to max_images)
            if image_count < max_images and sale['imageUrl']:
                download_image(sale['imageUrl'], sale['certNumber'])
                image_count += 1
            
            if image_count >= max_images:
                print(f"Reached maximum number of images ({max_images}) for this spec")
                break

def main():
    try:
        # Generate single output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"psa_sales_{timestamp}.csv"
        
        # Search for baseball cards
        for i in range(1, 5):
            search_results = search_psa_cards(page_number=i)
            specs = search_results['result']['data']['json']['search']['results']
        
        print(f"Found {len(specs)} total specs")
        
        # Process each spec
        for i, spec in enumerate(specs):
            spec_id = spec['specId']
            print(f"\nProcessing spec {spec_id}: {spec['collectibleDescription']} ({i+1}/{len(specs)})")
            
            try:
                # Fetch the data for this spec
                data = fetch_psa_data(spec_id)
                
                # Save to CSV and download up to 50 images
                save_to_csv(data, output_file, spec_info=spec, max_images=50, append=(i > 0))
                print(f"Data successfully appended to {output_file}")
                
            except Exception as e:
                print(f"Error processing spec {spec_id}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
