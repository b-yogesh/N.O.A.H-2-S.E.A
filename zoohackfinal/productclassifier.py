def difference(filepath):
    import requests
    r = requests.post(
        "https://api.deepai.org/api/image-similarity",
        files={
            'image1': open(filepath, 'rb'),
            'image2': open('/Users/rajaathota72/PycharmProjects/zoohackfinal/animal/Purse.jpeg', 'rb'),
        },
        headers={'api-key': 'ab5c1d28-f58a-4c0d-af22-983e0a0453aa'}
    )
    x = r.json()
    y = x['output']
    diff = y['distance']
    return diff
def path():
    filepath2 = '/Users/rajaathota72/PycharmProjects/zoohackfinal/animal/Purse.jpeg'
    import re
    c = re.split("/",filepath2)
    d = c[6]
    e = d[:]
    n1=e[:-5]
    return n1
def item(name):
    filepath3 = "/Users/rajaathota72/PycharmProjects/zoohackfinal/nonanimal/"+name+"jpeg"
    return filepath3
def main():
    import streamlit as st
    st.title("Get your Alternative Product _ Test case")
    import io
    file_buffer = st.file_uploader("Upload the image")
    text_io = io.TextIOWrapper(file_buffer)
    print(text_io)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.image(file_buffer,width = 700)
    if st.button("Compare"):
        result1= path()
        st.success("Match found and you have alternatives as product is from protected species")
        st.image("/Users/rajaathota72/PycharmProjects/zoohackfinal/nonanimal/Purse.jpeg", width=700)
        st.button("Click to shop now")
        st.success("You can also report anonymously for a cause")
if __name__ == "__main__":
    main()
