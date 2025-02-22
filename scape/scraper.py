import requests
import json
import csv
from datetime import datetime

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

def main():
    # Example spec_id - replace with your desired ID
    spec_id =  190786
    
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
