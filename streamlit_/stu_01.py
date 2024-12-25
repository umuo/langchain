import streamlit as st

st.title("Hello, World!")
st.write("This is a Streamlit app.")

with st.sidebar:
    st.write("第一个")
    st.write("第二个")
    st.write("第三个")

st.header("Header")
st.write("This is a header.")

def my_function():
    st.write("Button clicked!")

if st.button("登出", type="tertiary", key="my_custom_button"):
    my_function()

st.subheader("Subheader")
st.write("This is a subheader.")

st.caption("Caption")
st.write("This is a caption.")

st.code("def my_function():\n    print('Hello, World!')", language="python")

st.latex(r"e^{i\pi} - 1 = 0")

st.markdown("This is a Markdown text. **Bold** and _italic_.")

st.error("This is an error message.")

st.warning("This is a warning message.")

st.info("This is an info message.")

st.success("This is a success message.")

st.exception(ValueError("This is an exception message."))

st.json({"key": "value"})

st.table({"A": [1, 2], "B": [3, 4]})

st.metric("Metric", 0, 10)

st.progress(0.5)

st.balloons()

st.snow()

st.plotly_chart({"data": [{"x": [1, 2, 3], "y": [4, 5, 6]}], "layout": {"title": "Plotly Chart"}})

st.graphviz_chart("digraph G { a -> b; b -> c; a -> d; c -> d; }")

