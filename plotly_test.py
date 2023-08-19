import plotly.express as px
import pandas as pd

df = pd.DataFrame(dict(
    x = [1, 2, 3, 4],
    y = [1, 2, 3, 4]
))
#fig = px.line(df, x="x", y="y", title="Unsorted Input") 
#fig.show()

df = df.sort_values(by="x")
# px.plot(data, auto_open=False)
fig = px.line(df, x="x", y="y", title="Sorted Input") 
fig.write_image("figure.png", engine="kaleido")