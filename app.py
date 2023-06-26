
import pickle
import streamlit as st

from zipfile import ZipFile


# loading the temp.zip and creating a zip object
with ZipFile("regression_walmart_rf.pkl.zip", 'r') as zObject:
  
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall()

# loading the trained model
pickle_in = open('regression_walmart_rf.pkl', 'rb')
regressor = pickle.load(pickle_in)

@st.cache_data()

# defining the function which will make the prediction using the data which the user inputs
def prediction(Store, Holiday, Temperature, Fuel_Price, CPI, Unemployment, Day, Week,Month,Year):
  Holiday_Flag = 0  
  if Holiday == "Holiday":
      Holiday_Flag = 1
  else:
      Holiday_Flag = 0
    # Making predictions
  prediction = regressor.predict(
        [[Store, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Day, Week,Month,Year]])

  return prediction


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:white;text-align:center;"> Gali@ Pragyan AI and FUEL Walmart Sale Prediction ML App</h1>
    </div>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
  st.image("""https://cdn.corporate.walmart.com/dims4/WMT/0b04aa6/2147483647/strip/true/crop/2400x1260+0+0/resize/1200x630!/quality/90/?url=https%3A%2F%2Fcdn.corporate.walmart.com%2F6f%2Fd3%2Ff3f5a16f44a88d88b8059defd0a9%2Foption-signage.jpg""")

    # following lines create boxes in which user can enter data required to make prediction
    Store = st.number_input("EnterNumber of Store Number",min_value=1, max_value=50)
    Holiday = st.selectbox('Holiday Status',("Holiday","Not Holiday"))
    Temperature = st.number_input("Enter The Temperature value", min_value=1.0, max_value=75.0)
    Fuel_Price = st.number_input("Enter The Fuel_Price value", min_value=1.0, max_value=75.0)
    CPI = st.number_input("Enter The CPI value", min_value=100.0, max_value=250.0)
    Unemployment = st.number_input("Enter Unemployment Rate", min_value=1.0, max_value=20.0)
    Day = st.number_input("Enter The Day of Week",min_value=0, max_value=6)
    Week = st.number_input("Enter The Week of the Year",min_value=1, max_value=53)
    Month = st.number_input("Enter The Month of the Year",min_value=1, max_value=12)
    Year =  st.number_input('Enter The Year',min_value=2010, max_value=2023)
    result =""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Store, Holiday, Temperature, Fuel_Price, CPI, Unemployment, Day, Week,Month,Year)
        st.success('Your Walmart Sale Prediction is {}'.format(result))
        print(result)

if __name__=='__main__':
    main()
