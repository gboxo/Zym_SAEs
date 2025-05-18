import py3Dmol
import requests
import os
# import random # No longer needed
# from Bio.PDB import PDBParser # No longer needed for visualization part
import json # Needed for loading colors
import argparse # For command-line arguments

def download_pdb(pdb_id, filename):
    """Downloads a PDB file from the RCSB PDB database."""
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return filename
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        print(f"Attempting to download {pdb_id}.pdb from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {pdb_id}.pdb successfully.")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDB {pdb_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

# def generate_random_hex_color(): # No longer needed
#     """Generates a random hex color code."""
#     return f'#{random.randint(0, 0xFFFFFF):06x}'

def load_residue_colors(json_file):
    """Loads residue colors from a JSON file."""
    if not os.path.exists(json_file):
        print(f"Error: Color file {json_file} not found.")
        print(f"Please run 'generate_residue_colors.py' first.")
        return None
    try:
        with open(json_file, 'r') as f:
            residue_colors = json.load(f)
        print(f"Loaded colors for {len(residue_colors)} residues from {json_file}.")
        return residue_colors
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file}: {e}")
        return None
    except IOError as e:
        print(f"Error reading color file {json_file}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading colors: {e}")
        return None

def visualize_protein_with_loaded_colors(pdb_file, color_data, output_html="protein_colored_visualization.html"):
    """Visualizes a protein using colors loaded from a dictionary."""
    # No BioPython parsing needed here anymore
    # residue_colors = {} # Loaded externally now

    try:
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, 'pdb')

        # Set initial style (e.g., cartoon)
        view.setStyle({'cartoon': {'colorscheme': 'whiteCarbon'}}) # Default style

        # Apply loaded color to each residue
        # The keys in color_data are strings like "ChainID_ResSeq"
        for residue_key, color in color_data.items():
            try:
                chain_id, res_seq_str = residue_key.split('_')
                res_seq = int(res_seq_str)
                # py3Dmol selector for specific residue: {chain: 'A', resi: 123}
                view.setStyle({'chain': chain_id, 'resi': res_seq}, {'cartoon': {'color': color}})
            except ValueError:
                print(f"Warning: Skipping invalid residue key format '{residue_key}' in color file.")
            except Exception as e:
                 print(f"Warning: Error applying style for residue {residue_key}: {e}")

        view.zoomTo()

        # Save to HTML file
        view.write_html(output_html)
        print(f"Colored visualization saved to {output_html}")
        print(f"You can open this file in your web browser to view the structure.")

        # If running in Jupyter, uncomment the next line to display directly
        # view.show()

    except FileNotFoundError:
        print(f"Error: PDB file {pdb_file} not found for py3Dmol.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PDB file using pre-generated residue colors from a JSON file.")
    parser.add_argument("pdb_id", help="The PDB ID to visualize (e.g., 1B2Y).")
    parser.add_argument("-c", "--color_file", help="Input JSON color filename.", default=None)
    parser.add_argument("-o", "--output", help="Output HTML filename for visualization.", default="protein_colored_visualization.html")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    pdb_filename = f"/home/woody/b114cb/b114cb23/boxo/kl_divergence/{pdb_id}.pdb"
    color_json_filename = args.color_file if args.color_file else f"{pdb_id}_colors.json"
    output_html_filename = args.output

    # 1. Ensure PDB file exists (download if necessary)
    downloaded_pdb_file = download_pdb(pdb_id, pdb_filename)

    if downloaded_pdb_file:
        # 2. Load colors from JSON
        residue_colors = load_residue_colors(color_json_filename)

        if residue_colors:
            # 3. Visualize the protein with loaded colors
            visualize_protein_with_loaded_colors(downloaded_pdb_file, residue_colors, output_html_filename) 