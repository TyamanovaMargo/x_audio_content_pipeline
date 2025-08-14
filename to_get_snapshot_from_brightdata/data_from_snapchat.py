import requests
import csv

# Your existing snapshot ID
snapshot_id = "s_me7622tk2680muawiv"

url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
headers = {
    "Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d"
}

# Simple download parameters
params = {
    "format": "csv",
    "compress": False
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    # First save the full data
    with open("twitter_profiles_full.csv", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # Now extract only external links
    external_links = []
    
    # Read the CSV and extract external links
    csv_reader = csv.DictReader(response.text.splitlines())
    
    for row in csv_reader:
        # Check for external link in different possible column names
        external_link = (row.get('external_link') or 
                        row.get('url') or 
                        row.get('website') or 
                        row.get('profile_external_link') or
                        row.get('bio_link'))
        
        if external_link and external_link.strip():
            external_links.append({
                'username': row.get('username', '') or row.get('screen_name', ''),
                'profile_name': row.get('profile_name', '') or row.get('name', ''),
                'external_link': external_link.strip()
            })
    
    # Save only external links to a separate file
    if external_links:
        with open("twitter_external_links_only.csv", "w", encoding="utf-8", newline='') as f:
            fieldnames = ['username', 'profile_name', 'external_link']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(external_links)
        
        print(f"External links saved successfully!")
        print(f"Found {len(external_links)} profiles with external links")
        
        # Print the links
        for link_data in external_links:
            print(f"{link_data['username']}: {link_data['external_link']}")
    else:
        print("No external links found in the data")
        
else:
    print(f"Error downloading: {response.status_code} - {response.text}")
