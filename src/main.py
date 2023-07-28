import os
import sys
import warnings

import plotly.graph_objects as go

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.models import load_model

from crypto_analyst import CryptoAnalyst, LSTMModel
from data_collector import DataCollator

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import plotly.graph_objects as go


def plot_data(crypto, data) -> None:
    # Create the layout object with the title
    layout = go.Layout(
        title=go.layout.Title(
            text=f"{crypto} prices this month",
            font=dict(color="white", size=64, family="Arial"),
            x=0.05,
            y=0.95,
        ),
        xaxis_title="Date",
        yaxis_title="Price (GBP)",
        template="plotly_dark",
        width=1920,
        height=1080,
        font=dict(family="Arial", size=20),
        margin=dict(l=200, r=100, t=150, b=100),
        yaxis=dict(tickprefix="£", ticklen=10),
        xaxis=dict(showgrid=False, nticks=10),
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=data["Date"].iloc[i],
                y0=data["Close"].iloc[i],
                x1=data["Date"].iloc[i],
                y1=0,
                line=dict(color="yellow", width=1, dash="dot"),
            )
            for i in range(len(data["Date"]))
        ],
    )

    # Plot the data with the layout
    fig = go.Figure(
        data=go.Scatter(x=data["Date"], y=data["Close"], mode="lines"), layout=layout
    )

    # Make the plotted line thicker
    fig.update_traces(line=dict(width=3))

    # Add blinking green dot at the end of the line
    fig.add_trace(
        go.Scatter(
            x=[data["Date"].iloc[-1]],
            y=[data["Close"].iloc[-1]],
            mode="markers",
            marker=dict(color="green", size=20, symbol="circle-open"),
        )
    )

    # Add 20% buffer to the y-axis
    fig.update_yaxes(range=[min(data["Close"]) * 0.8, max(data["Close"]) * 1.2])

    # Remove the legend
    fig.update_layout(showlegend=False)

    # Move y ticks to the left a bit
    fig.update_yaxes(
        tickvals=fig.layout.yaxis.tickvals,
        tickmode="array",
        ticktext=fig.layout.yaxis.ticktext,
        tickangle=0,
    )

    # Add axis lines
    fig.update_xaxes(showline=True, linewidth=2, linecolor="white")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="white")

    # Save the plot as png
    fig.write_image(f"images/{crypto}.png")


def main():
    """Main function"""
    crypto = "MAT"
    start_date, end_date = "10-06-2023", "10-07-2023"
    data_collector = DataCollator(crypto, start_date, end_date)
    # data_collector = DataCollector(crypto)
    data = data_collector.load_crypto_data()
    plot_data(crypto, data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
