# Agents Directives

## Required Packages
You will need these packages in your environment to test the script

- numpy
- pandas
- librosa
- opensmile

## How to Test
`smile.wav` is available to test against, and its 'pretty scores' should be similar to the following:

- Focus: 47.7%
- Calming: 29.9%
- Sleep: 22.4%
- Other: 0%

Some small amount of deviation (~5%) across the full percentage is normal. For example this would also be acceptable:

- Focus: 44.7%
- Calming: 29.9%
- Sleep: 22.4%
- Other: 3.0%

Note that Focus dropped by 3% and Other increased by 3%. Since this is less than 5% across all categories this is acceptable result.

The Python script should export the following files:

- features.csv
- features.json
- scores.csv
- scores.json

Scores should contain the same data as the Focus, Calming, etc. 

Features should include the more precise features of the audio file.

You can see `example_features.csv`,`example_scores.csv`,`example_features.json`,`example_scores.json` for example results of my own run against smile.wav

## Example Usage

default: processes ./smile.wav if present

```python main.py```

process one file

```python main.py path\to\song.wav```

process a folder recursively

```python main.py path\to\folder```

process a folder non-recursively
```python main.py path\to\folder --no-recursive```
