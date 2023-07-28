import plotly.graph_objects as go
import plotly.io as pio


class Plotter:
    @staticmethod
    def plot_results(test_data):
        # Assuming you have the data in a pandas DataFrame called 'data'
        fig = go.Figure()

        # filter data to only include actual closing price and predicted closing price
        predicted_data = test_data[test_data["Type"] != "Actual"]
        actual_data = test_data[test_data["Type"] != "Predicted"]
        # Add the actual closing price
        fig.add_trace(
            go.Scatter(
                x=actual_data["Date"],
                y=actual_data["Close"],
                name="Actual Closing Price",
            )
        )

        # thickens the lines and change marker size
        fig.update_traces(line=dict(width=2), marker=dict(size=5))

        # Add the predicted closing price
        fig.add_trace(
            go.Scatter(
                x=predicted_data["Date"],
                y=predicted_data["Close"],
                name="Predicted Closing Price",
            )
        )

        # Get this screen size
        screen_size = (1920, 1080)
        # Customize the layout
        fig.update_layout(
            title="Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Closing Price",
            template="plotly_dark",
            # size is this screen size
            width=screen_size[0],
            height=screen_size[1],
        )
        # change marker symbol and colour and size
        fig.update_traces(
            mode="markers+lines",
            marker=dict(symbol="diamond-open", color="white", size=10),
        )
        # Save plot to png
        pio.write_image(fig, "images/MAT.png")
