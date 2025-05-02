# Index Arbitrage Trading Strategy

This project explores a PCA-based index arbitrage trading strategy.  
Developed as part of a group project during the Le Wagon Data Science and AI Bootcamp, I contributed by implementing PCA and developing the trading strategy.

---

## Project Description
- Utilized Principal Component Analysis (PCA) to identify stocks whose behavior closely mirrors the index (e.g., FTSE100), creating a replicated portfolio.
- Designed and backtested a trading strategy leveraging z-score analysis, assuming mean reversion in the spread between the replicated portfolio and the index.
- Developed a dynamic, user-friendly website using Streamlit to showcase the results, allowing users to explore and interact with key findings through customizable inputs.

---

## Detailed Project Structure

### **Data Input**
- Supported Indexes: FTSE100, S&P500, Nasdaq (optional to extend to additional indexes).
- **Data Requirements**:
  - The project folder must contain the data in `.csv` files stored within the `/data/` subfolder.
  - **CSV Format**:
    - Date format: `yyyy-mm-dd`.
    - Column structure:
      ```python
      rename_dict = {
          'Unnamed: 0': 'date',
          '1. open': 'open',
          '2. high': 'high',
          '3. low': 'low',
          '4. close': 'close',
          '5. volume': 'volume'
      }
      ```
    - Output: Dataframe with `date` as the index and columns representing the stocks used for PCA analysis.
  - Initial Data Coverage: From February 2022 to today.

---

### **Using PCA to Create Replicated Portfolio**
- **Input**:
  - Dataframe with:
    - **Index**: Dates.
    - **Columns**: Stocks used for PCA.
- **Process**:
  - Calculate logarithmic returns based on the frequency of the input dataframe.
  - Apply PCA to identify eigenvectors explaining variance.
  - Select top `X` stocks by weight based on the sum of eigenvectors.

---

### **Backtesting the Strategy**
This step evaluates the replication portfolio's performance and the trading strategy using historical data. The process involves:

- **Input**:
  - A dataframe containing:
    - **Rows**: Dates.
    - **Columns**: Stock weights in the replication portfolio.
    - **Values**: The weights of respective stocks for each date.
  - Stock prices and the target index (e.g., FTSE100) loaded from the `/data/` folder.

- **Portfolio Construction**:
  - Build the replication portfolio by applying the stock weights to historical stock prices.
  - Align data by dropping any dates or stocks not present in the weights dataframe.

- **Spread Calculation**:
  - Compute the spread as the difference between the index's returns and the replication portfolio's returns, monitored over a 60-day rolling window.

- **Z-Score Analysis**:
  - Calculate the z-score to measure the spread's deviation from its mean in terms of standard deviations.

- **Mean Reversion Trading Rules**:
  - **Buy Signal**: Triggered when the spread is 2 standard deviations below the mean (`z-score < -2`).
  - **Sell Signal**: Triggered when the spread is 2 standard deviations above the mean (`z-score > 2`).
  - **Exit Rule**: Close positions when the spread reverts towards the mean (`|z-score| < 0.5`).

---

### **Web Interface**
- Built with **Streamlit** to provide an interactive and dynamic user experience.
- Features:
  - Customizable inputs for testing different trading strategies and visualizing results.
  - Easy-to-navigate interface showcasing the performance of the replication portfolio and mean reversion signals.

---

## How to Run the Application Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Ilse-hutten/index-arbitrage
cd index-arbitrage
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Backend API (FastAPI with Uvicorn)

In a **new terminal** window:

```bash
uvicorn api.main:app --reload
```

> Make sure your FastAPI app is located in `api/main.py` and has the app instance defined as:
> ```python
> app = FastAPI()
> ```

### 4. Run the Streamlit Frontend

In another terminal:

```bash
streamlit run app/app.py
```



