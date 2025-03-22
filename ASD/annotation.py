import pandas as pd
import requests
from time import sleep

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", sep=",")  # Adjust delimiter if needed
species_list = df.columns[1:].tolist()  # Assuming first column is metadata
print(f"Extracted {len(species_list)} species from the dataset.")


# Function to fetch microbial function from NCBI
def get_ncbi_function(species_name):
    """Fetches functional description of microbes from NCBI Taxonomy API."""
    formatted_name = species_name.replace("_", " ")  # Convert to NCBI-friendly format
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    search_params = {
        "db": "taxonomy",  # Use taxonomy instead of genome
        "term": formatted_name,
        "retmode": "json"
    }

    try:
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Extract taxonomy ID if found
        if search_data["esearchresult"]["idlist"]:
            tax_id = search_data["esearchresult"]["idlist"][0]

            # Fetch taxonomic details
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {
                "db": "taxonomy",
                "id": tax_id,
                "retmode": "json"
            }

            summary_response = requests.get(summary_url, params=summary_params, timeout=10)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            if tax_id in summary_data["result"]:
                tax_info = summary_data["result"][tax_id]
                description = tax_info.get("description", "No function found")

                return {"Species": species_name, "Function": description}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching NCBI data for {species_name}: {e}")

    return {"Species": species_name, "Function": "Not Found"}


# Annotate species
annotations = []
for idx, species in enumerate(species_list, 1):
    print(f"Processing {idx}/{len(species_list)}: {species}")
    annotations.append(get_ncbi_function(species))
    sleep(1)  # Pause to avoid rate-limiting

# Convert to DataFrame and save
annotated_df = pd.DataFrame(annotations)
output_file = "microbial_functions_ncbi.csv"
annotated_df.to_csv(output_file, index=False)

print(f"Microbial function annotations saved to {output_file}")
