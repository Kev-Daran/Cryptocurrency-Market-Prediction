import streamlit as st
import requests

# Function to get news for the selected cryptocurrency
def get_crypto_news(symbol, api_key):
    news_url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}'
    response = requests.get(news_url)
    news_data = response.json()
    return news_data.get('articles', [])

# Streamlit web app
def main():
    st.title('Cryptocurrency News App')

    # Choose cryptocurrencies to get news
    selected_crypto = st.selectbox('Select Cryptocurrency', ['bitcoin', 'ethereum', 'litecoin', 'ripple', 'cardano'])

    # Your News API key
    api_key = '28899316ba0e47d4b55df8d21c9326b4'

    # Get news and display
    news_data = get_crypto_news(selected_crypto, api_key)
    st.subheader(f'Latest News for {selected_crypto.capitalize()}')

    if not news_data:
        st.write(f'No news available for {selected_crypto.capitalize()} at the moment.')
    else:
        for news_item in news_data:
            st.markdown(f"**{news_item['title']}**")
            st.write(news_item['description'])
            st.write(f"Source: {news_item['source']['name']}")
            st.markdown(f"[Read More]({news_item['url']})")
            st.write("---")  # Divider between news items

if __name__ == '__main__':
    main()
