import streamlit as st
st.title("This is a title text")
st.title('_This_is :blue[a title] :speech_balloon:')

#LaTeX
st.title('$E = mc^2$')

st.header('This is a header')
st.subheader('This is a sub header')

st.text('This is plain text with no formatting')

st.markdown('# This is a header\n **This is bold text** \n- This is a list item')

st.write('This is plain text using st.write')

data = {"Name": "Alice", "Age":30, "Occupation":"Engineer"}
st.write(data)

# write_stream - real time display of our data 
# button state false until pressed - nested button(making nested button true outer button becomes false)




