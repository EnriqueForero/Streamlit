# Libreria principal para hacer la aplicación Web. 
import streamlit as st
# Libreria para manipular datos. 
import pandas as pd
# Libreria para hacer cálculos numéricos
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Para correr el programa, en un terminal se escribe lo siguiente: 
# streamlit run app.py

#DATA_URL = ("/home/cicada/Downloads/rhyme/streamlit-sentiment/Tweets.csv")

st.title("Sentiment Analysis of Tweets about US Airlines")
# Título para la barra lateral. 
st.sidebar.title("Sentiment Analysis of Tweets")
# Incluyendo una anotación que va en la página principal. 
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦🦜")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")

# Esto es para que los datos sólo se carguen una vez. Va a ser persistente en el Caché. 
@st.cache(persist=True)
# Función para cargar los datos que vamos a utilizar. 
def load_data():
    data = pd.read_csv("Tweets.csv")
    # COnviertiendo una columna en tipo de fecha. 
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

# Cargando los datos. 
data = load_data()

# Para hacer pruebas. Puedo imprimir los datos. 
# st.write(data)

# Subtítulo en la barra lateral. 
st.sidebar.subheader("Show random tweet")
# Creando las opciones de selección de sentimientos. EL título es Sentiment. Las opciones 'positive', 'neutral', 'negative' 
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
# Esta es la funcionalidad para mostrar de forma aleatoria los tweets. 
# Lo mostramos en la barra lateral. Hacemos un query. La columna a utilizar es 
# airline_sentiment que debe ser igual la opción a la que selecciona el usuario 
# "random_tweet". Necesario utilizar "@" para acceder a la variable seleccionada.
# La columna donde está el texto es "text". Sólo mistramos un Tweet de forma aleatoria sample(n=1)
# Para sólo devolver texto y no un DataFrame indicamos la "celda" iat[0, 0], es decir la fila y la columna que en este caso es la primera (0). 
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# Creando otro subtítulo. 
st.sidebar.markdown("### Number of tweets by sentiment")
# Crea un selectbox con dos opciones. key='1' es para utilizar esto para diferentes visualizaciones. 
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
# Conteo de los Tweets según sentimiento. 
# Puedo imprimir la tabla con: st.write(sentiment_count)

# Creo un DataFrame. Primera Columna es Sentimen, utiliza el Index. La segunda columna el el conteo de Tweets. 
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

# Dejamos el checkbox hide por default. "If not".  Hide True. 
if not st.sidebar.checkbox("Hide", True):
    # Texto a mostrar
    st.markdown("### Number of tweets by sentiment")
    # Lo que se debe hacer si seleccionan "Bar plot"
    if select == 'Bar plot':
        # Configurando la imagen a mostrar. Los valores en X y en Y. Los colores van de acuerdo a los Tweets. height=500 son los pixeles. 
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        # Esto es para mostrar la gráfica. 
        st.plotly_chart(fig)
    # Else para indicar la otra opción que tenemos. 
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

# Puede mostrar los datos en un mapa con st.map(data) 
# Se requiere tener la latitud y la altitud. 
        
st.sidebar.subheader("When and where are users tweeting from?")
# Creamos un slider que va de cero a 23. Otra forma de hacer es así: 
# hour = st.sidebar.number_input("Hour to look at", min_value=1, max_value=24)
hour = st.sidebar.slider("Hour to look at", 0, 23)
# Esto me permite ver los Tweets de acuerdo a la hora seleccionada. Utilizamos la fecha de los Tweets. 
modified_data = data[data['tweet_created'].dt.hour == hour]
# Creando checkbox que permite ver o no la visualización. La visualización es hiden by default. Toca indicar key = 1. 
if not st.sidebar.checkbox("Close", True, key='1'):
    # Mensaje
    st.markdown("### Tweet locations based on time of day")
    # Muestra los valores. 
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    # Mostramos el dataframe con los datos seleccionados para gradicarlos en el mapa. 
    st.map(modified_data)
    # Si se selecciona mostrar los datosm entonces, mostrarlos. 
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2'):
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each airline")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)


@st.cache(persist=True)
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count


st.sidebar.subheader("Breakdown airline by sentiment")
# Vamos a comparar más de una aerolínea a la vez. 
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
# Para que no salga ningún error, indicamos que la elección debe ser mayor a cero. 
if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
    # Cada figura debe tener un nombre diferente. En este caso fig_3
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
if len(choice) > 0:
    # Me muestra los datos que están en Choice. isin(choice)
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',
        # facet_col='airline_sentiment' es para unir varios gráficos por esa variable. 
        # Colocamos los lables pero le cambiamos el nombre a tweets labels={'airline_sentiment':'tweets'
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'}, 
                          height=600, width=800)
    # Esto es para mostrar la figura creada
    st.plotly_chart(fig_0)

#     
st.sidebar.header("Word Cloud")
# Los botones de selección. 
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True, key='3'):
    # %s es para imprimir el sentimiento seleccionado. 
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    # Uniendo los comentarios. 
    words = ' '.join(df['text'])
    # Removiendo cosas que no nos dicen nada como 'http' and '@' y 'RT'
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    # QUitando las palabnras que no dicen nada stopwords=STOPWORDS
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
