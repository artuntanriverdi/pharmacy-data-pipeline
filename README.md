# Pharmacy Data Pipeline

This project implements a data pipeline for a pharmacy, built using **Pandas**, **NumPy**, and **OpenPyXL**. The pipeline processes Excel files containing inventory and financial data, performing calculations such as total stock quantities, average prices, and more. The results are saved into a new Excel file with summaries for further analysis.

## Features

- Reads Excel files (`stoklar.xlsx`, `cari.xlsx`, `giderler.xlsx`)
- Merges stock and financial data
- Calculates:
  - Total stock quantity
  - Average stock price
  - Total revenue
  - Total expenses
  - Net profit
- Generates summary sheets in a new Excel file (`summary.xlsx`)

## Requirements

- Python 3.7+
- pandas
- numpy
- openpyxl

