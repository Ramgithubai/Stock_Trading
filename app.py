#app.py
import os
import sys
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import EasyBridge AI modules
from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_ORDER
from src.llm.models import LLM_ORDER, get_model_info
from src.tools.api import get_price_data
from src.backtester import Backtester

# Load environment variables
load_dotenv()

# Define default values
DEFAULT_TICKERS = "AAPL,MSFT,NVDA"
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_MARGIN_REQUIREMENT = 0.5
DEFAULT_MODEL = "gpt-4o-mini"  # Default to a cheaper model

def validate_api_keys():
    """Check if required API keys are set"""
    missing_keys = []
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return "Warning: No API keys found. You must set at least one API key (OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY) in your .env file."
    
    available_apis = []
    if os.environ.get("GROQ_API_KEY"):
        available_apis.append("Groq")
    if os.environ.get("OPENAI_API_KEY"):
        available_apis.append("OpenAI") 

        
    return f"Available API providers: {', '.join(available_apis)}"

def format_portfolio_output(result, tickers, portfolio):
    """Format the output from the EasyBridge AI for display"""
    if not result or not result.get("decisions"):
        return "No decisions available."
    
    output = []
    
    # Format decisions
    output.append("# Trading Decisions")
    for ticker in tickers:
        if ticker not in result.get("decisions", {}):
            continue
            
        decision = result["decisions"][ticker]
        output.append(f"## {ticker}")
        output.append(f"**Action**: {decision.get('action', 'HOLD').upper()}")
        output.append(f"**Quantity**: {decision.get('quantity', 0)}")
        output.append(f"**Confidence**: {decision.get('confidence', 0):.1f}%")
        output.append(f"**Reasoning**: {decision.get('reasoning', 'No reasoning provided')}")
        output.append("")
    
    # Format analyst signals
    output.append("# Analyst Signals")
    for ticker in tickers:
        output.append(f"## {ticker}")
        ticker_signals = []
        
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
                
            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)
            
            ticker_signals.append(f"**{agent_name}**: {signal_type} ({confidence:.1f}%)")
        
        if ticker_signals:
            output.append("\n".join(ticker_signals))
        else:
            output.append("No signals available")
        output.append("")
    
    # Format portfolio summary
    output.append("# Portfolio Summary")
    output.append(f"**Initial Cash**: ${portfolio.get('cash', 0):,.2f}")
    output.append(f"**Margin Requirement**: {portfolio.get('margin_requirement', 0) * 100:.1f}%")
    
    # Calculate total value of current positions
    total_position_value = 0
    for ticker, position in portfolio.get("positions", {}).items():
        long_shares = position.get("long", 0)
        short_shares = position.get("short", 0)
        
        # We'd need current prices to calculate accurate position values
        # This is simplified and would need price data in a real implementation
        if long_shares > 0 or short_shares > 0:
            output.append(f"**{ticker}**: {long_shares} shares (long), {short_shares} shares (short)")
    
    return "\n".join(output)

def plot_portfolio_history(df, title="Portfolio Value"):
    """Create a plot of portfolio value over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Portfolio Value"], 'b-')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()
    return plt

def run_single_hedge_fund(
    ticker_input,
    selected_analyst_displays,
    model_display_name,
    initial_cash,
    margin_requirement,
    show_reasoning,
    start_date=None,
    end_date=None
):
    """Run the EasyBridge AI with the provided parameters"""
    # Parse tickers
    tickers = [t.strip() for t in ticker_input.split(",")]
    
    # Convert analyst display names to keys
    selected_analysts = []
    # Create the mapping from display names to keys
    analyst_display_to_key = {display: value for display, value in ANALYST_ORDER}
    for display in selected_analyst_displays:
        if display in analyst_display_to_key:
            selected_analysts.append(analyst_display_to_key[display])
    
    # Get the actual model name from the display name
    model_name = None
    model_provider = None
    for display, value, provider in LLM_ORDER:
        if display == model_display_name:
            model_name = value
            model_provider = provider
            break
    
    # Default to a Groq model when model or provider is missing
    if not model_name or not model_provider:
        model_name = "llama-3.3-70b-versatile"
        model_provider = "Groq"
    
    # Use today as the end date if not specified
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    # Use 3 months ago as the start date if not specified
    if not start_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    
    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": float(initial_cash),
        "margin_requirement": float(margin_requirement),
        "positions": {
            ticker: {
                "long": 0, 
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            } for ticker in tickers
        }
    }
    
    # Run the EasyBridge AI
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    
    # Format the output
    output = format_portfolio_output(result, tickers, portfolio)
    
    return output

def run_backtester(
    ticker_input,
    selected_analyst_displays,
    model_display_name,
    initial_cash,
    margin_requirement,
    show_reasoning,
    start_date,
    end_date
):
    """Run the backtester with the provided parameters"""
    # Parse tickers
    tickers = [t.strip() for t in ticker_input.split(",")]
    
    # Convert analyst display names to keys
    selected_analysts = []
    # Create the mapping from display names to keys
    analyst_display_to_key = {display: value for display, value in ANALYST_ORDER}
    for display in selected_analyst_displays:
        if display in analyst_display_to_key:
            selected_analysts.append(analyst_display_to_key[display])
            
    # Get the actual model name from the display name
    model_name = None
    model_provider = None
    for display, value, provider in LLM_ORDER:
        if display == model_display_name:
            model_name = value
            model_provider = provider
            break
    
    # Default to a Groq model when model or provider is missing
    if not model_name or not model_provider:
        model_name = "llama-3.3-70b-versatile"
        model_provider = "Groq"
    
    # Create and run the backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_cash),
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=float(margin_requirement),
    )
    
    # Run the backtest
    backtester.run_backtest()
    
    # Get performance data
    performance_df = backtester.analyze_performance()
    
    # Create the chart
    fig = plot_portfolio_history(performance_df)
    
    # Calculate summary statistics
    if len(performance_df) > 0:
        final_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = ((final_value - float(initial_cash)) / float(initial_cash)) * 100
        
        # Calculate additional metrics if we have enough data
        if len(performance_df) > 3:
            performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()
            sharpe_ratio = np.sqrt(252) * (performance_df["Daily Return"].mean() / performance_df["Daily Return"].std())
            
            # Calculate max drawdown
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
        else:
            sharpe_ratio = "N/A"
            max_drawdown = "N/A"
        
        summary = (
            f"# Backtest Results\n\n"
            f"**Initial Capital**: ${float(initial_cash):,.2f}\n"
            f"**Final Portfolio Value**: ${final_value:,.2f}\n"
            f"**Total Return**: {total_return:.2f}%\n"
            f"**Sharpe Ratio**: {sharpe_ratio if isinstance(sharpe_ratio, str) else sharpe_ratio:.2f}\n"
            f"**Max Drawdown**: {max_drawdown if isinstance(max_drawdown, str) else max_drawdown:.2f}%\n\n"
        )
    else:
        summary = "# No backtest results available"
    
    return summary, fig

def create_app():
    # Get analyst choices
    analyst_choices = [display for display, value in ANALYST_ORDER]
    
    # Create reverse mapping from display name to analyst key
    analyst_display_to_key = {display: value for display, value in ANALYST_ORDER}
    
    # Get model choices - use the display names directly
    model_display_options = []
    for display, value, _ in LLM_ORDER:
        model_display_options.append(display)
    
    # Define the interface
    with gr.Blocks(title="EasyBridge AI") as app:
        gr.Markdown("# EasyBridge AI")
        gr.Markdown("An AI-powered Fund Managing Application")
        
        api_status = gr.Markdown(validate_api_keys())
        
        with gr.Tab("Run Trading Decision"):
            with gr.Row():
                with gr.Column(scale=2):
                    ticker_input = gr.Textbox(
                        label="Tickers (comma-separated)",
                        placeholder="AAPL,MSFT,NVDA",
                        value=DEFAULT_TICKERS
                    )
                    
                    analysts = gr.CheckboxGroup(
                        choices=analyst_choices,
                        label="Select Analysts",
                        value=["Ben Graham", "Warren Buffett", "Technical Analyst", "Fundamentals Analyst"]
                    )
                    
                    # Set default to a Groq model
                    model = gr.Dropdown(
                        choices=model_display_options,
                        label="Select LLM Model",
                        value="[groq] llama-3.3 70b"  # Changed default to Groq
                    )
                    
                    with gr.Row():
                        initial_cash = gr.Number(
                            label="Initial Cash",
                            value=DEFAULT_INITIAL_CAPITAL
                        )
                        margin_req = gr.Number(
                            label="Margin Requirement",
                            value=DEFAULT_MARGIN_REQUIREMENT
                        )
                    
                    show_reasoning = gr.Checkbox(
                        label="Show Agent Reasoning",
                        value=False
                    )
                    
                    run_button = gr.Button("Run Trading Decision", variant="primary")
                
                with gr.Column(scale=3):
                    output = gr.Markdown("Results will appear here after running the model.")
                    
            # Create a wrapper function that includes the model_choices dictionary
            def run_hedge_fund_wrapper(*args):
                return run_single_hedge_fund(*args)
                
            run_button.click(
                fn=run_single_hedge_fund,
                inputs=[
                    ticker_input,
                    analysts,
                    model,
                    initial_cash,
                    margin_req,
                    show_reasoning
                ],
                outputs=output
            )
            
        with gr.Tab("Run Backtest"):
            with gr.Row():
                with gr.Column(scale=2):
                    bt_ticker_input = gr.Textbox(
                        label="Tickers (comma-separated)",
                        placeholder="AAPL,MSFT,NVDA",
                        value=DEFAULT_TICKERS
                    )
                    
                    bt_analysts = gr.CheckboxGroup(
                        choices=analyst_choices,
                        label="Select Analysts",
                        value=["Ben Graham", "Warren Buffett", "Technical Analyst", "Fundamentals Analyst"]
                    )
                    
                    # Set default to a Groq model
                    bt_model = gr.Dropdown(
                        choices=model_display_options,
                        label="Select LLM Model",
                        value="[groq] llama-3.3 70b"  # Changed default to Groq
                    )
                    
                    with gr.Row():
                        bt_initial_cash = gr.Number(
                            label="Initial Cash",
                            value=DEFAULT_INITIAL_CAPITAL
                        )
                        bt_margin_req = gr.Number(
                            label="Margin Requirement",
                            value=DEFAULT_MARGIN_REQUIREMENT
                        )
                    
                    with gr.Row():
                        bt_start_date = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d")
                        )
                        bt_end_date = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            value=datetime.now().strftime("%Y-%m-%d")
                        )
                    
                    bt_show_reasoning = gr.Checkbox(
                        label="Show Agent Reasoning",
                        value=False
                    )
                    
                    bt_run_button = gr.Button("Run Backtest", variant="primary")
                
            with gr.Row():
                bt_summary = gr.Markdown("Backtest results will appear here.")
                bt_plot = gr.Plot()
                
            bt_run_button.click(
                fn=run_backtester,
                inputs=[
                    bt_ticker_input,
                    bt_analysts,
                    bt_model,  # Pass the model dropdown directly
                    bt_initial_cash,
                    bt_margin_req,
                    bt_show_reasoning,
                    bt_start_date,
                    bt_end_date
                ],
                outputs=[bt_summary, bt_plot]
            )
            
        with gr.Tab("About"):
            gr.Markdown("""
            # About AI EasyBridge AI
            
            EasyBridge AI is a startup based on Chennai. The goal of this project is to explore the use of AI to make trading decisions. 
            EasyBridge Fund Managers by Vivek, Sheriff , Ramachandran & Dharani Babu
            
            ## System Components
            
            This system employs several agents working together:
            
            1. **Ben Graham Agent** - The godfather of value investing, only buys hidden gems with a margin of safety
            2. **Bill Ackman Agent** - An activist investor, takes bold positions and pushes for change
            3. **Cathie Wood Agent** - The queen of growth investing, believes in the power of innovation and disruption
            4. **Warren Buffett Agent** - The oracle of Omaha, seeks wonderful companies at a fair price
            5. **Charlie Munger Agent** - Warren Buffett's partner, only buys wonderful businesses at fair prices
            6. **Valuation Agent** - Calculates the intrinsic value of a stock and generates trading signals
            7. **Sentiment Agent** - Analyzes market sentiment and generates trading signals
            8. **Fundamentals Agent** - Analyzes fundamental data and generates trading signals
            9. **Technicals Agent** - Analyzes technical indicators and generates trading signals
            10. **Risk Manager** - Calculates risk metrics and sets position limits
            11. **Portfolio Manager** - Makes final trading decisions and generates orders
            
            ## Disclaimer
            
            This project is for **educational and research purposes only**.
            
            - Not intended for real trading or investment
            - No warranties or guarantees provided
            - Past performance does not indicate future results
            - Creator assumes no liability for financial losses
            - Consult a financial advisor for investment decisions
            """)
            
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(debug=True)