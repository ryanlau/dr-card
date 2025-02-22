import requests
import json
import csv
from datetime import datetime
import os

def hex_encode_payload(spec_id):
    # Create the payload dictionary
    payload = {
        "specId": spec_id,
        "page": 1,
        "pageSize": 500
    }
    
    # Convert to JSON string and encode to hex
    json_str = json.dumps(payload)
    hex_str = json_str.encode('utf-8').hex()
    return hex_str

def fetch_psa_data(spec_id):
    hex_encoded = hex_encode_payload(spec_id)
    url = f"https://www.psacard.com/api/psa/trpc/auctionPrices.itemDetailAuctionResultsTable?input=%22{hex_encoded}%22"
    print(url)
    
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
    
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded image for cert {cert_number}")
    except Exception as e:
        print(f"Failed to download image for cert {cert_number}: {str(e)}")

def save_to_csv(data, output_file):
    # Extract the individual sales from the response
    sales = data['result']['data']['json']['spec']['historicalAuctionInfo']['pastListingInfo']['individualSales']
    
    # Define CSV headers
    headers = [
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
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for sale in sales:
            writer.writerow({
                'auctionItemId': sale['auctionItemId'],
                'auctionHouse': sale['auctionHouse'],
                'imageUrl': sale['imageUrl'],
                'listingType': sale['listingType'],
                'dateOfSale': sale['dateOfSale'],
                'grade': sale['grade'],
                'salePrice': sale['salePrice'],
                'certNumber': sale['certNumber']
            })
            # Download the image for this sale
            download_image(sale['imageUrl'], sale['certNumber'])

def main():
    # Example spec_id - replace with your desired ID
    spec_id =  190785
    
    try:
        # Fetch the data
        data = fetch_psa_data(spec_id)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"psa_sales_{spec_id}_{timestamp}.csv"
        
        # Save to CSV
        save_to_csv(data, output_file)
        print(f"Data successfully saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
