import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from solar_model import SolarProject # Assumes the class above is in solar_model.py

def sidebar_inputs():
    """Create and manage sidebar input widgets."""
    st.sidebar.header("Project Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        capacity_mw = st.number_input("Capacity (MW)", 50, 500, 100, 10)
        capex_per_watt = st.slider("CapEx ($/W)", 0.80, 2.00, 1.35, 0.05)
        capacity_factor = st.slider("Capacity Factor", 0.15, 0.35, 0.25, 0.01)

    with col2:
        ppa_price = st.slider("PPA Price ($/MWh)", 20, 80, 45, 1)
        opex_per_kw = st.slider("OpEx ($/kW-yr)", 10, 30, 17, 1)
        itc_rate = st.slider("ITC Rate", 0.0, 0.50, 0.30, 0.05)

    st.sidebar.header("Financing")
    debt_fraction = st.sidebar.slider("Debt Fraction", 0.0, 0.90, 0.70, 0.05)
    debt_rate = st.sidebar.slider("Debt Interest Rate", 0.03, 0.09, 0.06, 0.005)

    return {
        'capacity_mw': capacity_mw, 'capex_per_watt': capex_per_watt, 
        'opex_per_kw_year': opex_per_kw, 'capacity_factor': capacity_factor,
        'ppa_price': ppa_price, 'itc_rate': itc_rate, 
        'debt_fraction': debt_fraction, 'debt_rate': debt_rate
    }

def display_metrics(results, ppa_price):
    """Display the main financial metrics in columns."""
    st.header("Financial Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Levered IRR", f"{results['levered_irr']:.1%}" if results['levered_irr'] else "N/A",
                  help="Internal Rate of Return on the equity invested.")
    with col2:
        st.metric("NPV @ 10%", f"${results['npv_10pct']/1e6:.1f}M",
                  help="Net Present Value of equity cash flows, discounted at 10%.")
    with col3:
        delta_lcoe = f"{results['lcoe'] - ppa_price:.1f} vs PPA" if results['lcoe'] else None
        st.metric("LCOE", f"${results['lcoe']:.2f}/MWh" if results['lcoe'] else "N/A",
                  delta=delta_lcoe, delta_color="inverse",
                  help="Levelized Cost of Energy: The average revenue per MWh needed to break even.")
    with col4:
        st.metric("Payback Period", f"{results['payback_years']} years" if results['payback_years'] else "N/A",
                  help="The number of years it takes for the project to repay its initial equity investment.")

def plot_cash_flow(cf_df):
    """Generate and display the cash flow chart."""
    st.header("Cash Flow Analysis")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cf_df.index, y=cf_df['Revenue'], name='Revenue', marker_color='lightgreen'))
    fig.add_trace(go.Bar(x=cf_df.index, y=cf_df['OpEx'], name='OpEx', marker_color='lightcoral'))
    fig.add_trace(go.Scatter(x=cf_df.index, y=cf_df['Levered_FCF'], name='Levered Free Cash Flow',
                             mode='lines+markers', line=dict(color='blue', width=3)))
    fig.update_layout(title="Annual Cash Flows", xaxis_title="Year", yaxis_title="$", barmode='relative', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
def display_sensitivity_charts(project):
    """Run and display sensitivity analysis charts."""
    st.header("Sensitivity Analysis")
    sens_tab1, sens_tab2, sens_tab3 = st.tabs(["PPA Price", "CapEx", "Capacity Factor"])
    
    with sens_tab1:
        ppa_range = np.linspace(project.ppa_price * 0.8, project.ppa_price * 1.2, 7)
        ppa_sens = project.sensitivity_analysis('ppa_price', ppa_range)
        fig = px.line(ppa_sens, x='ppa_price', y='levered_irr', title='IRR vs. PPA Price')
        st.plotly_chart(fig, use_container_width=True)

    with sens_tab2:
        capex_range = np.linspace(project.capex_per_watt * 0.8, project.capex_per_watt * 1.2, 7)
        capex_sens = project.sensitivity_analysis('capex_per_watt', capex_range)
        fig = px.line(capex_sens, x='capex_per_watt', y='levered_irr', title='IRR vs. CapEx')
        st.plotly_chart(fig, use_container_width=True)

    with sens_tab3:
        cf_range = np.linspace(project.capacity_factor * 0.8, project.capacity_factor * 1.2, 7)
        cf_sens = project.sensitivity_analysis('capacity_factor', cf_range)
        fig = px.line(cf_sens, x='capacity_factor', y='levered_irr', title='IRR vs. Capacity Factor')
        st.plotly_chart(fig, use_container_width=True)

def display_tornado_chart(project, base_results):
    """Run and display the tornado chart for IRR drivers."""
    st.header("Tornado Chart - IRR Drivers")
    variables = {
        'ppa_price': (project.ppa_price * 0.9, project.ppa_price * 1.1),
        'capex_per_watt': (project.capex_per_watt * 0.9, project.capex_per_watt * 1.1),
        'capacity_factor': (project.capacity_factor * 0.9, project.capacity_factor * 1.1),
        'opex_per_kw_year': (project.opex_per_kw_year * 0.9, project.opex_per_kw_year * 1.1),
        'debt_rate': (project.debt_rate * 0.9, project.debt_rate * 1.1)
    }
    tornado_df = project.run_tornado_analysis(base_results, variables)
    
    if not tornado_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=tornado_df['variable'], x=tornado_df['low_impact'], name='Downside', orientation='h', marker_color='red'))
        fig.add_trace(go.Bar(y=tornado_df['variable'], x=tornado_df['high_impact'], name='Upside', orientation='h', marker_color='green'))
        fig.update_layout(title="Impact on IRR (±10% Variation)", xaxis_title="Change in IRR", barmode='relative', height=400)
        st.plotly_chart(fig, use_container_width=True)

# --- Main Application ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Utility-Scale Solar Financial Model", layout="wide")
    st.title("Utility-Scale Solar Project Financial Model")
    st.markdown("An interactive dashboard to explore the business case for a utility-scale solar project.")

    # 1. Get user inputs from the sidebar
    params = sidebar_inputs()

    # 2. Create a SolarProject instance and run the analysis
    project = SolarProject(**params)
    results = project.run_analysis()

    # 3. Display the key financial metrics
    display_metrics(results, params['ppa_price'])
    
    # 4. Show the cash flow chart
    plot_cash_flow(results['cash_flow_statement'])
    
    # 5. Display sensitivity and tornado charts
    display_sensitivity_charts(project)
    display_tornado_chart(project, results)
    
    # 6. Add a data download option
    st.sidebar.markdown("---")
    if st.sidebar.button("Export Cash Flows to CSV"):
        csv = results['cash_flow_statement'].to_csv().encode('utf-8')
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"solar_cashflows_{params['capacity_mw']}MW.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()