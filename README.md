# openSMILE-Focara

A small utility for classifying music tracks into **Focus**, **Calming**, **Sleep**, or **Other**. It combines
`librosa` for audio features with `openSMILE` functionals and exports both wide feature tables and easy to read
scores in CSV and JSON formats.

## Installation

1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (This pulls in packages such as `numpy`, `pandas`, `librosa`, and `opensmile`.)

## Usage

Process the default demo file `smile.wav`:
```bash
python main.py
```

Process an individual WAV file:
```bash
python main.py path/to/song.wav
```

Process a folder recursively:
```bash
python main.py path/to/folder
```

Process a folder without recursion:
```bash
python main.py path/to/folder --no-recursive
```

## Outputs

Running the script writes four files to the working directory:

- `scores.csv` / `scores.json` – percentage scores for Focus, Calming, Sleep, and Other.
- `features.csv` / `features.json` – a wide set of extracted audio features for each track.

Example output files for the bundled demo clip are provided as `example_*.csv` and `example_*.json`.

## Testing

Run the classifier on the included sample to verify the setup:
```bash
python main.py
```
It should report scores close to:

- Focus: ~47.7%
- Calming: ~29.9%
- Sleep: ~22.4%
- Other: ~0%

Minor variation (±5% total) is acceptable. Successful execution also creates
`scores.csv`, `scores.json`, `features.csv`, and `features.json`.

