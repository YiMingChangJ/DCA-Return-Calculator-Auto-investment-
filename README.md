# Automatic Investment Strategy Calculator

This project is an interactive **automatic investment (DCA) calculator** built with [Streamlit](https://streamlit.io/).  
It helps you explore how regular contributions, investment horizon, compounding frequency, and annual return affect your long-term wealth.

The core logic is implemented in an `Investment` class, and the Streamlit app provides a simple dashboard on top of it.

---

## Features

- 🔢 **Configurable inputs**
  - Investment per period (e.g., monthly contribution)
  - Investment duration in years
  - Investment frequency (times per year)
  - Annual interest rate (slider)
- 📋 **Detailed output**
  - Total principal invested
  - Total earnings
  - Total portfolio value (principal + earnings)
- 📈 **Investment growth graph**
  - Visualizes the growth of your portfolio over time
- 🧩 **Reusable class**
  - `Investment` class can be imported into other Python scripts for further analysis

---

## Example Dashboard

> Example: \$4,000 per period, 35 years, 12 investments per year, 12% annual return.

![Automatic Investment Dashboard](images/Interactive_dashboard_screenshot.png)

---

## Example Growth Curve

This graph shows the growth of **principal + earnings** (in millions) over the 35-year period for the same example parameters.

![Investment Growth Example](images/Investment_t=35_p=4000.0_a=12_r=12.0.jpg)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YiMingChangJ/DCA-Return-Calculator-Auto-investment-.git
cd DCA-Return-Calculator-Auto-investment-
