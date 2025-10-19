import numpy as np
import numpy_financial as npf
from scipy.optimize import newton
import pandas as pd
import matplotlib.pyplot as plt

class SolarProject:
    """
    A comprehensive financial model for a utility-scale solar project.
    
    This class encapsulates all the necessary parameters and calculations to
    assess the financial viability of a utility-scale solar power plant. It
    builds a pro-forma cash flow statement and calculates key financial
    metrics like IRR, NPV, LCOE, and payback period.
    """
    
    def __init__(self, 
                 capacity_mw=100,
                 capex_per_watt=1.35,
                 opex_per_kw_year=17,
                 capacity_factor=0.25,
                 degradation_rate=0.005,
                 ppa_price=45,
                 ppa_escalator=0.02,
                 project_life=30,
                 itc_rate=0.30,
                 debt_fraction=0.70,
                 debt_rate=0.06,
                 tax_rate=0.21):
        
        # Technical Parameters
        self.capacity_mw = capacity_mw  # Rated capacity of the solar farm in Megawatts (MW).
        
        # Assumption: Average capacity factor for a utility-scale solar project in the U.S.
        # Source: U.S. Energy Information Administration (EIA) data often shows national averages in the 24-26% range.
        self.capacity_factor = capacity_factor
        
        # Assumption: Annual degradation rate of solar panels. Modern panels degrade at about 0.5% per year.
        # Source: National Renewable Energy Laboratory (NREL) studies and manufacturer warranties.
        self.degradation_rate = degradation_rate
        
        # Assumption: The operational life of the project. Modern solar farms are often planned for a 30-year or longer lifespan.
        # Source: Lawrence Berkeley National Laboratory reports project life assumptions have increased to over 30 years.
        self.project_life = project_life

        # Cost Parameters
        # Assumption: Capital expenditure per watt (DC). This includes panels, inverters, racking, and all construction costs.
        # Note: As of late 2024/early 2025, utility-scale solar CapEx has seen fluctuations.
        # Source: NREL's Annual Technology Baseline and other industry reports.
        self.capex_per_watt = capex_per_watt
        
        # Assumption: Annual operating and maintenance (O&M) expenses per kilowatt.
        # Source: Berkeley Lab's "Benchmarking Utility-Scale PV Operational Expenses" suggests a range of $13-$25/kW-yr.
        self.opex_per_kw_year = opex_per_kw_year
        
        # Revenue Parameters
        # Assumption: The price per Megawatt-hour (MWh) in the first year of the Power Purchase Agreement (PPA).
        # This is a highly variable, market-dependent figure.
        # Source: LevelTen Energy's PPA Price Index, which tracks quarterly PPA prices. Q1 2025 prices averaged around $57/MWh in North America. We use a more conservative estimate.
        self.ppa_price = ppa_price
        
        # Assumption: The annual escalation rate of the PPA price.
        # This is typically set to hedge against inflation and is often in the 1-2.5% range.
        # Source: Industry standard PPA contract terms.
        self.ppa_escalator = ppa_escalator
        
        # Tax & Policy Parameters
        # Assumption: The federal Investment Tax Credit (ITC) rate. The Inflation Reduction Act of 2022 restored it to 30%.
        # Source: U.S. Department of Energy and IRS guidelines.
        self.itc_rate = itc_rate
        
        # Assumption: Federal corporate tax rate.
        # Source: The Tax Cuts and Jobs Act of 2017 set the rate at 21%.
        self.tax_rate = tax_rate
        
        # Assumption: 5-year MACRS (Modified Accelerated Cost Recovery System) depreciation schedule for solar assets.
        # Source: IRS Publication 946.
        self.macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
        
        # Financing Parameters
        # Assumption: The proportion of the project's capital cost financed by debt.
        # Project finance for renewables often uses high leverage.
        # Source: Industry reports on renewable energy project finance often cite a 60-80% debt fraction.
        self.debt_fraction = debt_fraction
        
        # Assumption: The interest rate on the debt.
        # This is sensitive to prevailing interest rates and project risk.
        # Source: Varies with the federal funds rate and market conditions for project finance.
        self.debt_rate = debt_rate
        
        # Calculated values
        self._update_calculated_values()
        
    def _update_calculated_values(self):
        """Helper method to update attributes that depend on initial parameters."""
        self.capacity_kw = self.capacity_mw * 1000
        self.total_capex = self.capacity_kw * 1000 * self.capex_per_watt # capex is per watt
        self.annual_generation_year1 = self.capacity_mw * 8760 * self.capacity_factor
        
    def calculate_annual_generation(self, year):
        """Calculates the energy generation for a given year, accounting for degradation."""
        # Generation decreases each year due to panel degradation.
        return self.annual_generation_year1 * (1 - self.degradation_rate) ** year
    
    def calculate_revenue(self, year):
        """Calculates the revenue for a given year based on generation and PPA price."""
        generation_mwh = self.calculate_annual_generation(year)
        # PPA price escalates annually.
        ppa_price_year = self.ppa_price * (1 + self.ppa_escalator) ** year
        return generation_mwh * ppa_price_year
    
    def calculate_opex(self, year):
        """Calculates operating expenses, assuming a fixed inflation rate."""
        # Assumption: OpEx escalates with inflation at 2.5% annually.
        # Source: A common assumption for long-term financial modeling.
        return self.capacity_kw * self.opex_per_kw_year * (1.025 ** year)
    
    def calculate_depreciation(self, year):
        """Calculates depreciation based on the 5-year MACRS schedule."""
        if year < len(self.macrs_schedule):
            # The depreciable basis is reduced by half of the ITC value.
            # This is a specific provision of the tax code for projects taking the ITC.
            depreciable_basis = self.total_capex * (1 - 0.5 * self.itc_rate)
            return depreciable_basis * self.macrs_schedule[year]
        return 0
    
    def calculate_debt_service(self):
        """Calculates the constant annual debt service payment (interest + principal)."""
        debt_amount = self.total_capex * self.debt_fraction
        
        # This is a standard annuity payment formula for an amortizing loan.
        if self.debt_rate > 0:
            annual_payment = debt_amount * (self.debt_rate * (1 + self.debt_rate) ** self.project_life) / \
                             ((1 + self.debt_rate) ** self.project_life - 1)
        else:
            annual_payment = debt_amount / self.project_life if self.project_life > 0 else 0
            
        return annual_payment
    
    def build_cash_flow_statement(self):
        """Builds a complete pro-forma cash flow statement for the project."""
        years = list(range(self.project_life + 1))
        cf = pd.DataFrame(index=years)
        
        # Year 0: Investment Period
        cf.loc[0, 'CapEx'] = -self.total_capex
        cf.loc[0, 'ITC'] = self.total_capex * self.itc_rate
        cf.loc[0, 'Debt_Drawn'] = self.total_capex * self.debt_fraction
        cf.loc[0, 'Equity_Invested'] = -(self.total_capex - cf.loc[0, 'Debt_Drawn'])
        
        annual_debt_payment = self.calculate_debt_service()
        
        # Operating Years (1 to project_life)
        for year in range(1, self.project_life + 1):
            # Note: year-1 is used to index from the first year of operation
            cf.loc[year, 'Revenue'] = self.calculate_revenue(year - 1)
            cf.loc[year, 'OpEx'] = -self.calculate_opex(year - 1)
            cf.loc[year, 'Depreciation'] = -self.calculate_depreciation(year - 1)
            
            # Simplified interest calculation (should ideally be on a schedule)
            debt_balance_start = self.total_capex * self.debt_fraction * (1 - (year-1)/self.project_life)
            interest_payment = debt_balance_start * self.debt_rate
            cf.loc[year, 'Interest'] = -interest_payment
            
            # Tax calculations
            ebt = (cf.loc[year, 'Revenue'] + cf.loc[year, 'OpEx'] + 
                   cf.loc[year, 'Depreciation'] + cf.loc[year, 'Interest'])
            cf.loc[year, 'Taxable_Income'] = ebt
            # Assumes no NOL carryforwards for simplicity. A real model would track this.
            cf.loc[year, 'Tax'] = -max(0, ebt) * self.tax_rate
            
            # Cash Flow Calculations
            cf.loc[year, 'Debt_Service'] = -annual_debt_payment

        cf.fillna(0, inplace=True)
        
        # Calculate Unlevered and Levered Free Cash Flow
        cf['Unlevered_FCF'] = (cf['Revenue'] + cf['OpEx'] + cf['Taxable_Income'] * (1 - self.tax_rate) - 
                                cf['Taxable_Income'] + cf['Depreciation'])
        cf.loc[0, 'Unlevered_FCF'] = cf.loc[0, 'CapEx'] + cf.loc[0, 'ITC']
        
        cf['Levered_FCF'] = (cf['Revenue'] + cf['OpEx'] + cf['Tax'] + 
                               cf.loc[1:, 'Interest'] + cf.loc[1:, 'Debt_Service'])
        cf.loc[0, 'Levered_FCF'] = cf.loc[0, 'Equity_Invested'] + cf.loc[0, 'ITC']

        return cf

    @staticmethod
    def calculate_irr(cash_flows):
        """Calculates the Internal Rate of Return (IRR) for a series of cash flows."""
        try:
            # The newton function finds the root of a function, which in this case is the rate where NPV is zero.
            return newton(lambda r: npf.npv(r, cash_flows), 0.1, maxiter=100)
        except (RuntimeError, ValueError):
            # If the IRR calculation fails to converge, return None.
            return None

    def calculate_lcoe(self, discount_rate=0.06):
        """Calculates the Levelized Cost of Energy (LCOE)."""
        # Present value of total capital and operating costs
        pv_costs = self.total_capex
        for year in range(self.project_life):
            opex = self.calculate_opex(year)
            pv_costs += opex / (1 + discount_rate) ** (year + 1)
        
        # The ITC directly reduces the net cost of the system.
        pv_costs -= self.total_capex * self.itc_rate
        
        # Present value of all energy generated over the project's life
        pv_generation_mwh = 0
        for year in range(self.project_life):
            generation = self.calculate_annual_generation(year)
            pv_generation_mwh += generation / (1 + discount_rate) ** (year + 1)
            
        return pv_costs / pv_generation_mwh if pv_generation_mwh > 0 else 0

    def run_analysis(self):
        """Runs the full financial analysis and returns a dictionary of key metrics."""
        cf_statement = self.build_cash_flow_statement()
        
        unlevered_fcf = cf_statement['Unlevered_FCF'].values
        levered_fcf = cf_statement['Levered_FCF'].values
        
        # Cumulative cash flow for payback period calculation
        cumulative_cf = np.cumsum(levered_fcf)
        payback_years = np.where(cumulative_cf > 0)[0]
        
        results = {
            'unlevered_irr': self.calculate_irr(unlevered_fcf),
            'levered_irr': self.calculate_irr(levered_fcf),
            'npv_10pct': npf.npv(0.10, levered_fcf),
            'lcoe': self.calculate_lcoe(),
            'payback_years': payback_years[0] if len(payback_years) > 0 else "N/A",
            'total_capex': self.total_capex,
            'annual_generation_year1': self.annual_generation_year1,
            'cash_flow_statement': cf_statement
        }
        return results

    def sensitivity_analysis(self, variable, values):
        """Performs a sensitivity analysis on a given project variable."""
        base_value = getattr(self, variable)
        results = []
        
        for value in values:
            setattr(self, variable, value)
            self._update_calculated_values() # Ensure dependent variables are updated
            
            analysis = self.run_analysis()
            results.append({
                variable: value,
                'levered_irr': analysis['levered_irr'],
                'lcoe': analysis['lcoe'],
                'npv_10pct': analysis['npv_10pct']
            })
            
        # Reset to the original value
        setattr(self, variable, base_value)
        self._update_calculated_values()
        
        return pd.DataFrame(results)

    def run_tornado_analysis(self, base_results, variables):
        """Calculates the impact of variable swings for a tornado chart."""
        tornado_data = []
        base_irr = base_results.get('levered_irr')
        if base_irr is None:
            return pd.DataFrame()

        for var, (low, high) in variables.items():
            original_value = getattr(self, var)

            # Low scenario
            setattr(self, var, low)
            self._update_calculated_values()
            low_irr = self.run_analysis()['levered_irr']
            
            # High scenario
            setattr(self, var, high)
            self._update_calculated_values()
            high_irr = self.run_analysis()['levered_irr']

            # Reset to original value for next iteration
            setattr(self, var, original_value)
            self._update_calculated_values()

            if low_irr is not None and high_irr is not None:
                tornado_data.append({
                    'variable': var.replace('_', ' ').title(),
                    'low_impact': low_irr - base_irr,
                    'high_impact': high_irr - base_irr,
                    'total_swing': abs(high_irr - low_irr)
                })

        return pd.DataFrame(tornado_data).sort_values('total_swing', ascending=True)