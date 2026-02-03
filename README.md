# Data Cleaner (Python)

A Python-based data cleaning utility that:
- Repairs malformed Excel files (CSV-in-a-cell)
- Normalizes names, emails, phones, and addresses
- Generates clean outputs and error reports
- Handles real-world messy data automatically

## Features
- Excel input (.xlsx)
- Automatic CSV expansion when data is stored in one column
- Whitespace, casing, and formatting normalization
- Email and phone validation
- Deduplication strategies
- Clean + error outputs

## Usage

```bash
python data_cleaner.py excel --path messy.xlsx --sheet Sheet1 --out cleaned --dedupe smart
