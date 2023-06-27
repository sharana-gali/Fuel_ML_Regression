
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
    st.image("""data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAM4AAAD1CAMAAAAvfDqYAAAAyVBMVEX////tOzsAAADNzc3tNzf4+Pj19fXtOTntNTX8/PywsLDt7e3c3NzDw8PHx8fy8vIVFRVERERVVVXh4eHT09Pn5+d8fHw+Pj6ampqmpqZ0dHQzMzPsMTHQ0NC0tLSIiIhpaWmTk5MfHx8pKSkPDw+Dg4MtLS1NTU0bGxtcXFxmZmZ4eHhJSUmgoKDuSEj+8fHsKir84uL5xMTzgoLwXFz72dnvTU34ubnxcHD1mJj85ub2qKj0jo7xZWX3tLTvV1f70ND3o6P1k5MGBovUAAARDElEQVR4nO2dZ2OruBKGMa7YuHfjCsSxnTjOSe/J2f//o67QjEAS4JINTrJX76ezCmA90mg0I4RW05SUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlJSUlL5Dxmbz9IfoabMxdl5Irnx62mSPVLWDZdxePb883FwQvT+8/L263Wy/8O7i4u71+vHt48929G/R5u36JjUcFjJ5okxhOEzdXL89RV14/wIXZuC64cXry30c+Tdp8/c9VSjkU5xIXVPvf0MN/3yTIsjChYVC6ub5BwFtnlOXYhUZ0eXFPQ9k3F/EXZj6MUC3D5eZcBVBmcJLYHF/XgrxF14+3CZf1VG/KKrfkq4w7u+GcXX0Gn74yqr58TqM6Blfw7v7xH3CTCdyfZH/6Es0zxexLY7VvPmgV169b8P2OujiOWkewyx2ujrTYroeib9o/B3uoEmlCncez9VFYdeFmWHYdSQgi+GMQ396vtxmPxzPHjTEMi+fj4BzwnBM+S8fW0dDwHNz9b4HDeHJfySP02M4JekPT697VZKY0d1Ok0Twh6ip92u1ZjhygPU3ahKJbvZ9r8v8TRynz3Ck8o992/wAZd4Tn35icLKPOzzvp5S8dysijS0WPyXQOZ7bSDraGSPOXCy+T6JzSBjxcSScrlh8nQhOaviSME4ZcepCqbEruvmkMu8JWxvDORVKN/sEBJ9QPvVxHJyGUHp7mQgNsbaEIx2GsxRKPxLDeUwWp4I4K6H0KjGcf5LFSSPO9L+FMzgSTsLGNkKcjlCa3NhJOAptIc5EKP2TGM79cXBmQulmv9TtYGXu/iSLYyJOTyg1EolAvQwu4ZCa4azF4oc9U9EDlfTQ8XGkNannRAZPfph0/tZEnKJYfLt3ynyICq9J5zvtaJxNItY2THztkOFIy2zGc+Hruydzl/haQQyOdvv+9b5t+JL4+7hSDI72+OXdk6ELwN+E8/Xdk3S85qmKOOXQX77aVyeeWG/HMfZbet5X+cur5Gm0XCwOmXu+0Nzyw3+O8UIkV4vF0d6+jidfuD7K+9HcIh4n+5j/Ku9WeE04lGY4djyOtrn+ouFTSH4CFXEqkX/dbH+DuzdN4qGnj+MATjrm7w//fjbNH61vCM58O87m+l+Pn8Lrx7FofJxR3AWbl71fw0Ur2HxwDBx8dS3vkOB4Hv/NgnV++Jr8G9EwTujFNafnzKfnn/zw4aj7cXLd3Tjax80nHUIm9XjcPXp74Wi3159y2IWLtyPvOGQ4ze2Xbf75xAAqXFwde7Mhw2nvuM54KxwYIeQvb47o0lC5OqVxd+EQg3s9aABl8i/fsClvfxzt6eWAAVRIPX/HRt3cOcVZ7IGjbf6m9jW44fsxkrWwEMfeB0czru722xw1PGJcIyh7SnGcvXDIALrZw+BIrnbMSIAX4sz3xNGernc6hEIq+Z1Rcco2KI61Lw4J4Xak3MO7tyQrvF0H4+za8Mp2u36PEKe7P46m3cc7OOIEjrMoEKPsEnDkvZNbdXsXM4Ay+eukKrqfEKd+EI52G51zZy4ev/mLEMQ5PwxH+xPl4I6w93uXEOf0QBxt8xCagDLDt++mYTiNQ3E0Q+bJFBLeM7CPPo2jGdfCmkjh4gfQaMYKcKqH3yqs8WR+Bk0Zlg2dyEXdHdoE/iCf+nYvoOXGUwtfIOiL7qB4sMERf4A0l98Xpvla6nPL6qKsuXt6cMJlwIcK+ULSO3D3UbrSMs0myjTT6cPtBT68ONLLmyPo46aQKnzDGkdSuroYpn6CU/sqXT18Y36jpKSk9J9Vrln5RAzwA2W00+PZ2bLuWJ/ICX6YBo6rMy1+P05vulo2EGee++7afIFypTasSuvd34djmON1r7cem/yw7wBO/dfhlDt1mm0u6ifcq1zEafwqHMOckDrXHNwR1Q0GPuIsf+ypPBFqzhzd6fTH5XF/4HkzN4Sz+kU4aTLeVyOocJUe6hGsayDO9PfgjC1dnwQAc2HjAOIMfg2O95n7hHNm3ofVwQ4vxOn8liCnXJPecGQtfhs7Hu9x8ktw2t6eLvEDlj7/+RTiTH4JztSrrFhkzrlPDxFn9jtw6FfUU7HMmHHfuSJO71fgwCsB6WMprWIHBxIgjvS12w9Vhe5VlXfbVVfBgQQTwJGJf6bgHJ/Qe3UzAESc8ClMP1DVgS7GAGEhzmfe6BxdZje6dzj9JpwRvE3btlMVcfivDsx0Jb2tBarkgnKlFZlS5Ezu1na6XDFll5kzR+TuUaTFkCdz5YZZESuSXuwcGDPACbbpl2Z1V68tx3Geu9kf1L3HOh15a7/RGvfO6r6TNIpLLyAR27LU75x7WYozle2hmi7OBue1wCc1Z8S2akvuujS8HRxo8UIcP4gz4eW1voj2dcYa9ld6mvPN1CyeLLs13ud3auG2LDb8u+110GDV8mzQsGjq4s8YJqb9dvAANDb5BKwtOK2gtlE7902vOsH6j2+is5pfxnz+VA/5zPZK5+U3WC8oc1mhuWBFjv8r6ArkL9zDOC7WvR20nnyUkaeyrS+mvfGaVdVhaWB/NR0sXb5CM/aYYIJuzYntkLsH2B42s8PWSecE764hfY5rnwH7lRL+bjd+nzf8bA1xOnzrhbxdxdWX9OeMHtaIP9ig1OUqRMD1+nTOX9Jy9NMiTauKePcJdze2/AJ+NDvg6uH6dsLaqBO7sgFX2HAHifDsWY+dcCgPudFc76CjyWJsZHF/NsBOqa2XiFnNmiXvwAAW7ZKxsGKtyqrF1aoJQ8UGyyLAVm9t42X+4RZjZoGxMRnY7Rz63dHrLUMboYnWRW9NfjBI8tJWuAdhmy/dYzHGOpB6n8EfiaEs/bubOEI5d4MLfg4dxM2Gfloy/E/ALXYjbrLR44MywOma8OPgVtEYbMEpVaesZp4MlsVyV+C+2DK1uylA2PgP0p380Ug4+LkiHLW0XY0edjs7VclvV3ZEWezkAw+ue2bQ7uJFOXQIggfp6yt+KuoDMr+ghRtJK96vWmBX2QlWeSzuWyqDHVmBtZUQx7uvadVgTistpbobpwzHLUZOjIBz6j1l7R/GhKECn6KatXNhQhzBd2X8LIk4xFzqOptUKi4N3ku2JUy6Ju4/DyYDrLlVonVi6ReeHRlkm+z4DuJyIlM0wGmQ7mwv/Q9d8PifKdegRal7q/VgpKBwI2lLa6LxerVc0ZP40tLgza2wiYPnIQ5xyobr316uyUZZ9HkCBx7CWbY9ewhcmR70GSrbkhpjKVdIW7H+OuGaswj/lI60ZTNC4OgRsGt4NfaLMTTgpsDsLOA5DUeWgLNqe0M9MJw6a+h44czAtfoKfyPbteO+1PQ1k60Icbw0uWv5FanCxOlwd1ZPAp7zUAXXzKxMvvOgrltrNZPbF8OaRrXsNnauPKAnCRxjFnDOPUvn3CXUfcEvalYnAU9X/v4YcAYl0k3c4IADQt3oj5X5S/jzmwBnacykU5CiNMbg2OdGnFPPD3GfrYLxSB8UcTy29Ikr4JxVNYf3XOjwtyV1OAVw3g+6dEUMZPcu+QrM78HSuIF9S+ZT/hhP6EVXytjGVgAk8gBOJ2sK52SZ0TjNfsNx7dN10/DPcjsJ7ABwBi3L0iLVLi7J3eczk9zSsoHdn3gQZ0maiXeCENa48mfGramPYwuogDPRJi5fdzzDRPTNpb4ftPdKjJjDOYOOLksHiKGq4zm7e9LW2k4czkzo27QTbfWlwOCW/PVrHAF1YR276oZxmrT1nX653LeI78iFcMD5DtaRmVKJDmq7R+4+1Zdm1orGmVaXp7wfacF14UGc9c//1SecowCcfnMhxM9VJ4RTomMVXGN7SiKeGJxpx4lYB4CIv1vxqkpcbaNUl3DQkZyZ4oTbtCKt3lPFH0DcXwGy2BfvwI+vORysDxpqrqN3bGiaoC1p+7vTRtS7Ivo3P6+c6QOI2qfyyJuUxYWadjcWR0uzs+dPgxkGcPpnujBV4BeXHA44Mj8o9dNWDoeas7uaRwTvEDX56VqVjeSp7BdnJ0uhb0tbcLQWS1mDegLOZCkeo5s7lS7L2nznEBXxUdywBxzLDU++BoVfBP4J959z8w7D6YrvMjA0jJkAw2cAU5za1BK9EaYIMjU3vvCbzBAOiSLDy3owaZ4HBVnsHu6BgNORvCmYSex8zhZN/B4FnK60tIg4fh9D8wqjdBbCwZJVaBI1BvKlLMjhEkKMq6Q4MQs4ccdblHBtyK8Y+jtbXEzISjiYbvErKDjvcPEMCytDAdsIHAtfpyZYGxedYUg7lSpCjcCNjYUxWPKfgzjSvgJMyn0cXCARroEbuYx1FipBYY4iVBSGOLeWgzjikaSaATixy9KYxPoJEeJIqR0aGzNZfBMhrryFKo/hYiiLz8KgcoXCejSOK63RoLHFB4EQcvsnTiOO5AkBxx+BmEUJx1SXJKv1F4VCTpXlzXxZdS73BeA40igBnFr8K51xFE5NMk7AcZjLhchJXHgrx+CEcz7TCjdG2o7G6UoDDzzbIn4TRzPK2M6lZVLAsVjNsOr8qiW6ZT65xmWHUEvi0bZTvgzXNDhnNwgza2zekf5HBrxKrtDQgLOS6gA4fkOz3Ia/BEYX9wIDqxjerIQ4Qt9C9M37xUH4Ig+HugxHixUEy/5zwOucSIu+EBX4CV0EDlrQghv463An8Dj8H9rhlbzw3OQJghwYdkbUCILe8Q28KLcSh9NgJog43KRHepV6fDuEI55lzeHwq6DjGr2b92PQX/LiM+CAS52vIlaiTNEYi5FPgRDGT4zGoQplBzb9fX6djaVOMTicmzcmtTN55AGO7OUhooaYzLIjItGi2BuAIwfB4FD8ZcNyqEIjZ0LTG4uLptay/Yg4taCk2T2jnqTG1R5w5OAMcMBBLIVEDeXZKHf6BcUJTX2Ac8ZGVMXWxT411nqThksWN0+sI1uG7gGiCsbnWK9QnAXX3IAjT/+Qvk3pvzu6HXpPZdZF26I4oV6E83/86MvEFxj+j7W7pwadHPlZZh3ZMsG49xMHo95tQ67N5RKQy8rTP3icDqvqifRnOl2ccUNqLJsM4FjCMMhhQOUXzEgYQXHqXHsBTsSZohjM+Z6kT9oT0lOuMTpSD/I44FIrYVxvoafBdxnF6coBOJw7F4wwnPXY18stb1hRHH6ZmF5Ui4jlx5ANzHEotr0VfsDhbAtw5Ggclj5gPJq67DfLlvymfyybDFXVFkc1zjLoWM2u985vJeNA6hQR/DJrA+NtLz3/RceOE9rQLePAexcwYC9KXPCP7zviawGGU5frQOda3o3iOjddByifU6ujCSUf0lCcyBOQsHPpVoJKg079FIc/wQZw5DshC4chVvXa1M+HjDKZG92ZNBUVI3FouMwHMIYNNdKthoX/j5Gp3K+9uN5hyax3d9eFWYSOJ5sbs+DZZE8MDQG9U6VPqffTpjkanxBOZyBYldFO4yht9EdNH7vUKoJtWbN0kz3f5JaFYUyje1iOvWuyzcoE1kids3XZbErxSCn4/xfq06qPQ2yl6C3x5swybjaoF1v8nh58pj5Jt3Nk8tU51Rozyet4cb9bI/IetWAjhSZ4tNhrP7+HWmzLRrcPiBPdmTuOY+vzCnZyjT1Ld+WY1mQbBawemEef3e01/UznKuIvndA5x8VSh1SvPZ4su45tz7urSXEUsup2uZJmqpTZU3LlSiVcSjDHg7nrLPuspJ1uUY0q5MEt7lHpSqVckT1utdyxXKfRYy/xSpUR3k3G8ii4m9zL6lkSqkftKlstUVW/Yhd7znvQp+/Oenf/ih20SkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkpKSkr/N/of81BT7Fm4NDYAAAAASUVORK5CYII=""")

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
