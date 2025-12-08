# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:13:36 2025

@author: 92412
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Investment class (using your provided class code)
class Investment():
    def __init__(self, price: float, years: int, times: int, interest: float, print_value=False, graph_bool=False, save=False) -> None:
        self.price = price
        self.years = years
        self.times = times
        self.interest = interest
        self.print = print_value
        self.graph = graph_bool
        self.save = save

    def Auto_Investment_calculator(self) -> float:
        if self.times < 1 or self.years < 1:
            raise ValueError
        total_assets = 0
        total_assets_list = []
        interest_rate_period = self.interest / self.times
        for i in range(self.years):
            yearly_investment = 0
            if self.times == 1:
                yearly_investment = self.price
                total_assets += yearly_investment
                total_assets = total_assets * (1 + self.interest)
            else:
                for j in range(1, self.times + 1):
                    yearly_investment += self.price * (1 + interest_rate_period) ** j
                total_assets = total_assets * (1 + self.interest)
                total_assets += yearly_investment
                total_assets_list.append(total_assets / 1e6)

        if self.print:
            st.write(f"Investment Duration: {self.years} years")
            st.write(f"Interest Rate: {self.interest * 100}% annually")
            st.write(f"Investment Frequency: {self.times} times per year")
            st.write(f"Each Investment: ${round(self.price)}")
            st.write(f"Total Principal: ${round(self.price * self.times * self.years / 1e6, 3)} million")
            st.write(f"Total Earnings: ${round((total_assets - self.price * self.times * self.years) / 1e6, 2)} million")
            st.write(f"Total Investment (Principal + Earnings): ${round(total_assets / 1e6, 2)} million")
        
        if self.graph:
            self.plot_investment_growth(total_assets_list)

        return total_assets

    def plot_investment_growth(self, total_assets_list):
        size = 26
        textsize = 18
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams.update({'font.size': size})
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(total_assets_list, linestyle='-', marker='o', markerfacecolor='r', markeredgecolor='k', markersize=4)
        ax.set_xlabel('Years', fontsize=size)
        ax.set_ylabel('Principal and Earnings ($M)', fontsize=size)

        st.pyplot(fig)

# Streamlit UI
def create_investment_dashboard():
    st.title("Automatic Investment Strategy Calculator")
    
    price = st.number_input("Investment per period ($)", min_value=100, value=4000)
    years = st.number_input("Investment Duration (Years)", min_value=1, value=35)
    times = st.number_input("Investment Frequency (Times per Year)", min_value=1, value=12)
    interest_rate = st.slider("Annual Interest Rate (%)", min_value=0, max_value=30, value=12) / 100

    print_details = st.checkbox("Print Investment Details", value=True)
    show_graph = st.checkbox("Show Investment Growth Graph", value=True)
    save_graph = st.checkbox("Save Graph", value=False)

    # Calculate total earnings when user presses the button
    if st.button("Calculate Total Earnings"):
        investment = Investment(price, years, times, interest_rate, print_value=print_details, graph_bool=show_graph, save=save_graph)
        total_earnings = investment.Auto_Investment_calculator()
        st.write(f"Total Earnings after {years} years: ${total_earnings:,.2f}")
        
# Run the dashboard
if __name__ == "__main__":
    create_investment_dashboard()

# streamlit run "c:/Users/92412/Desktop/Projects Development/1. Investment/DCA_auto_investment_Project/Interactive_dashboard.py"