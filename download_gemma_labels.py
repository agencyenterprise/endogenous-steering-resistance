"""Download and convert Gemma SAE labels from Neuronpedia."""

import gzip
import json
import csv
import urllib.request
from pathlib import Path

def download_and_convert_labels(model_id, layer_id, output_file):
    """
    Download labels from Neuronpedia and convert to CSV format.

    Args:
        model_id: e.g., "gemma-2-9b-it"
        layer_id: e.g., "9-gemmascope-res-16k"
        output_file: path to output CSV file
    """
    base_url = f"https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model_id}/{layer_id}/explanations"

    # Collect all labels
    all_labels = {}

    # Download all batch files
    batch_num = 0
    while True:
        url = f"{base_url}/batch-{batch_num}.jsonl.gz"

        try:
            print(f"Downloading batch {batch_num}...")
            with urllib.request.urlopen(url) as response:
                compressed_data = response.read()

            # Decompress and parse
            decompressed_data = gzip.decompress(compressed_data).decode('utf-8')

            for line in decompressed_data.strip().split('\n'):
                if line:
                    entry = json.loads(line)
                    index = int(entry['index'])
                    description = entry['description']
                    entry_id = entry['id']

                    all_labels[index] = {
                        'label': description,
                        'uuid': entry_id,
                        'status': 'public',  # Assuming all are public
                        'notes': entry.get('notes', '')
                    }

            batch_num += 1

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"No more batches (last was batch-{batch_num-1})")
                break
            else:
                raise

    # Write to CSV
    print(f"Writing {len(all_labels)} labels to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index_in_sae', 'label', 'status', 'notes', 'uuid'])
        writer.writeheader()

        # Write in order
        for index in sorted(all_labels.keys()):
            writer.writerow({
                'index_in_sae': index,
                'label': all_labels[index]['label'],
                'status': all_labels[index]['status'],
                'notes': all_labels[index]['notes'] or '',
                'uuid': all_labels[index]['uuid'],
            })

    print(f"Done! Saved {len(all_labels)} labels to {output_file}")
    return len(all_labels)

if __name__ == "__main__":
    # Gemma 2B labels (layer 16, 16k SAE)
    download_and_convert_labels(
        model_id="gemma-2-2b",
        layer_id="16-gemmascope-res-16k",
        output_file="data/labels/gemma-2-2b-res-16k-layer-16.csv"
    )

    # Gemma 27B labels (layer 22, 131k SAE)
    download_and_convert_labels(
        model_id="gemma-2-27b",
        layer_id="22-gemmascope-res-131k",
        output_file="data/labels/gemma-2-27b-res-131k-layer-22.csv"
    )
