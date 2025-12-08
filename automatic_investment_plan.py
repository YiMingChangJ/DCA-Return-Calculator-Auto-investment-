# -*- coding: utf-8 -*-
"""
This module provides tools to model and calculate the earnings from an automatic investment strategy. 
The `Investment` class encapsulates the logic for estimating returns based on user-specified parameters 
such as investment frequency, amount, annual yearly_return rate, and investment duration.

The model assumes compound yearly_return and allows users to visualize the investment growth over time. 
It offers features to calculate the total earnings, display results, and generate graphs of the investment 
trajectory, making it a practical tool for financial planning.

Good luck with your investment!
"""
from __future__ import annotations



import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import matplotlib.pyplot as plt

class Investment():
    """
    The `Investment` class models an automatic investment strategy and calculates total 
    earnings over a specified period, accounting for compound yearly_return. Users can customize 
    the investment frequency, input amount, and annual return rate. The class also provides 
    options to print investment details and visualize growth through graphs.

    Attributes:
        price (float): The amount invested at each interval (e.g., weekly, monthly, annually).
        years (int): The total number of years for the investment.
        times (int): The number of investments per year (frequency).
        yearly_return (float): The annual average return rate (as a decimal, e.g., 0.12 for 12%).
        print (bool): Whether to display investment details.
        graph (bool): Whether to generate a graph of the investment growth.

    Raises:
        ValueError: Raised if the number of years or investment frequency is less than 1.
    """
    def __init__(self,init:float, price:float, years:int, times:int, yearly_return:float, print_value = False, graph_bool = False, save = False) -> None:
        self.init = init
        self.price = price
        self.years = years
        self.times = times
        self.yearly_return = yearly_return
        self.print = print_value
        self.graph = graph_bool
        
    def Auto_Investment_calculator(self) -> Union[float]:
        """
        Calculates the total earnings from an automatic investment strategy.

        This method computes the total investment and earnings using compound yearly_return 
        based on the specified parameters (price, years, times, and yearly_return). It allows 
        users to visualize the growth trajectory through graphs and provides detailed 
        investment insights.

        Returns:
            Union[float]: The total earnings (principal + return) after the specified period.

        Process:
            - Validates input parameters (e.g., number of years and frequency).
            - Computes the annual and cumulative returns using compound yearly_return.
            - Optionally prints detailed investment metrics.
            - Optionally generates and displays a graph of the investment growth.

        Raises:
            ValueError: If the investment years or frequency is less than 1.

        Example:
            >>> Total_earnings = Investment(4000, 35, 12, 0.12, True, True, False).Auto_Investment_calculator()
        """     
        
        'Set up variables'
        total_assets  = self.init # total investment including earnings/return
        total_assets_list = [] # list to store total investment
        yearly_return_rate_period = self.yearly_return/self.times # compound yearly_return rate: Annually/Monthly/Weekly
        
        if self.times < 1 or self.years < 1:
            raise ValueError
        
        
        for i in range(self.years): # years loop
            yearly_investment = 0
            if self.times == 1:
                yearly_investment = self.price
                total_assets += yearly_investment # Principal + annual return rate

                total_assets = total_assets *(1+self.yearly_return) 
            else:
                for j in range(1,self.times+1):
                    yearly_investment += self.price*(1+yearly_return_rate_period)**j  # monthly yearly return
                total_assets = total_assets * (1+self.yearly_return)
                total_assets += yearly_investment # Principal + annual return rate
                total_assets_list.append(total_assets/1e6)
        
        if self.print == True:
            print("Length of Investment: ", years,'years')
            print("Average Annual Reture Rate: ",self.yearly_return*100,'%')
            print("Investment Frequency: ", times)
            print("Each Investment: ($)", round(price))
            print("Annual Investment Amount: ($) ", yearly_amount)
            print("Princial: ($)", round(price*times*years/1e6,3),' millions')
            print("Return: ($)", round((total_assets-price*times*years)/1e6,2),' millions')
            print("Principal and Earnings: ($)",round(total_assets/1e6,2), ' millions')
            # print(total_assets_list)
        
        if self.graph == True:
            size = 26
            textsize = 26
            plt.rcParams['lines.linewidth'] = 3
            plt.rcParams.update({'font.size': size})
            plt.rc('xtick', labelsize=size)
            plt.rc('ytick', labelsize=size)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            
            fig, ax = plt.subplots(figsize=(8, 5))            
            ax.plot(total_assets_list, linestyle='-', marker='o', markerfacecolor='r', markeredgecolor='k', markersize=4)
            ax.text(0.01,total_assets_list[-2],'amount = {}'.format(self.price),fontsize = textsize)
            ax.text(0.01,total_assets_list[-3],'times = {}'.format(self.times),fontsize = textsize)

            ax.text(0.01,total_assets_list[-4],r'r = {} \%'.format(self.yearly_return*100),fontsize = textsize)
            # ax[1,0].text(0.03,lim_y-0.65,r'$\eta = {}$'.format(eta2),fontsize = textsize)
            # ax[1,0].text(0.02,lim_y-0.45,r'$\tilde\alpha_\mathrm m = {}$'.format(alpha_m2),fontsize = textsize)

            ax.set_xlabel('Years',fontsize=size)
            ax.set_ylabel(r'Principal and Earnings (\$M)',fontsize=size)
            
            if save == True:
                fig.savefig('Investment_t={}_p={}_a={}_r={}.jpg'.format(self.years,self.price,self.times,self.yearly_return*100),format='jpg',dpi=1200,bbox_inches='tight')
            
            plt.show()

        return total_assets 

years = 29 # number of years investment
initial_investment = 200000
yearly_return = 0.21 # annual yearly_return with principal/capital
times = int(12) # Investment frequency or number of investments
price = 3200 # automatic invest amount for every time (weekly, monthly, yearly)
yearly_amount = price*times

print_value = True
graph = True
save = False
Total_earnings = Investment(initial_investment,price,years,times,yearly_return,print_value,graph,save).Auto_Investment_calculator()

#%%

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Investment class (using your provided class code)
class Investment():
    def __init__(self, price: float, years: int, times: int, yearly_return: float, print_value=False, graph_bool=False, save=False) -> None:
        self.price = price
        self.years = years
        self.times = times
        self.yearly_return = yearly_return
        self.print = print_value
        self.graph = graph_bool
        self.save = save

    def Auto_Investment_calculator(self) -> float:
        if self.times < 1 or self.years < 1:
            raise ValueError
        total_assets = 0
        total_assets_list = []
        yearly_return_rate_period = self.yearly_return / self.times
        for i in range(self.years):
            yearly_investment = 0
            if self.times == 1:
                yearly_investment = self.price
                total_assets += yearly_investment
                total_assets = total_assets * (1 + self.yearly_return)
            else:
                for j in range(1, self.times + 1):
                    yearly_investment += self.price * (1 + yearly_return_rate_period) ** j
                total_assets = total_assets * (1 + self.yearly_return)
                total_assets += yearly_investment
                total_assets_list.append(total_assets / 1e6)

        if self.print:
            st.write(f"Investment Duration: {self.years} years")
            st.write(f"yearly_return Rate: {self.yearly_return * 100}% annually")
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
    yearly_return_rate = st.slider("Annual yearly_return Rate (%)", min_value=0, max_value=30, value=12) / 100

    print_details = st.checkbox("Print Investment Details", value=True)
    show_graph = st.checkbox("Show Investment Growth Graph", value=True)
    save_graph = st.checkbox("Save Graph", value=False)

    # Calculate total earnings when user presses the button
    if st.button("Calculate Total Earnings"):
        investment = Investment(price, years, times, yearly_return_rate, print_value=print_details, graph_bool=show_graph, save=save_graph)
        total_earnings = investment.Auto_Investment_calculator()
        st.write(f"Total Earnings after {years} years: ${total_earnings:,.2f}")
        
# Run the dashboard
if __name__ == "__main__":
    create_investment_dashboard()

# streamlit run c:/users/92412/Desktop/Investment/auto_investment/automatic_investment_plan.py
