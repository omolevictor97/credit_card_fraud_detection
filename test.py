from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle

with open("model.pkl", "rb") as file:
    loaded_file = pickle.load(file)

scaler = StandardScaler()

st.title("Credit Card Fraud Detection")


def main():
    OWN_CAR_AGE = st.number_input("Enter your car age:", min_value=0, max_value=50)
    FLAG_MOBIL = st.number_input("Did client provide mobile phone (1=YES, 0=NO)")
    FLAG_EMP_PHONE = st.number_input("Did you provide your work phone contact")
    FLAG_CONT_MOBILE = st.number_input("Was mobile phone reachable (1=YES, 0=NO)")
    REGION_RATING_CLIENT = st.number_input("Our rating of the region where client lives (1,2,3)")
    REGION_RATING_CLIENT_W_CITY = st.number_input("Our rating of the region where client lives with taking city into account (1,2,3)")
    EXT_SOURCE_1 = st.number_input("Normalized score from external data source", min_value =0.0, max_value=1.0)
    BASEMENTAREA_AVG = st.number_input("Normalized information about building where the client lives", min_value=0.0, max_value=1.0)
    YEARS_BUILD_AVG = st.number_input("Years of house building",min_value=0.0, max_value=1.0)
    COMMONAREA_AVG = st.number_input("Normalized Common Area",min_value=0.0, max_value=1.0)
    ENTRANCES_AVG = st.number_input("Normalized Entrance Average",min_value=0.0, max_value=1.0)
    #COMMONAREA_AVG = st.number_input("Normalized information about building where the client lives", min_value=0.0, max_value=1.0)
    FLOORSMIN_AVG = st.number_input("Normalized Floors min avg", min_value=0.0, max_value=1.0)
    LANDAREA_AVG = st.number_input("Normalized Land Area avg",min_value=0.0, max_value=1.0)
    #LANDAREA_AVG = st.number_input("Normalized information about building where the client lives", min_value=0.0, max_value=1.0)
    LIVINGAPARTMENTS_AVG = st.number_input("Normalized Living apartment avg",min_value=0.0, max_value=1.0)
    NONLIVINGAPARTMENTS_AVG = st.number_input("Normalized Non-Living apartment avg",min_value=0.0, max_value=1.0)
    NONLIVINGAREA_AVG = st.number_input("Normalized non-living area avg",min_value=0.0, max_value=1.0)
    COMMONAREA_MODE = st.number_input("Common area mode", min_value=0, max_value=1)
    LANDAREA_MODE = st.number_input("Land area mode", min_value=0.0, max_value=1.0)
    LIVINGAPARTMENTS_MODE = st.number_input("Living apartment mode",min_value=0.0, max_value=1.0)
    BASEMENTAREA_MEDI = st.number_input("Basement Area",min_value=0.0, max_value=1.0)
    TOTALAREA_MODE = st.number_input("Total area mode",min_value=0.0, max_value=1.0)
    FLAG_DOCUMENT_3 = st.number_input("Document", min_value=0.0, max_value=1.0)
    NAME_CONTRACT_TYPE_Cash_loans = st.number_input("Contract product type", min_value=0, max_value=1)
    CODE_GENDER_F = st.number_input("Gender (1 for F) and (0 for M)", min_value=0, max_value=1)
    NAME_TYPE_SUITE_Unaccompanied = st.number_input("Where you accompanied to your house (1 for Yes) and (0 for No)", min_value=0, max_value=1)
    NAME_INCOME_TYPE_Working = st.number_input("Do you have income from working (1 for Yes) and (0 for No)", min_value=0, max_value=1)
    EDUCATION_TYPE = st.number_input("Highest Education (1 for secondary) and (0 for others)", min_value=0, max_value=1)
    FAMILY_STATUS = st.number_input("Are you married (1 for yes) and (0 for No)", min_value=0, max_value=1) 
    HOUSING_TYPE = st.number_input("Housing Type (1 for apartment) and (0 for others)", min_value=0, max_value=1)
    HOUSETYPE_MODE = st.number_input("Housing Type mode (1 for block of flats) and (0 for others)")

    features = [
        OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_CONT_MOBILE, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY,
        EXT_SOURCE_1, BASEMENTAREA_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG,ENTRANCES_AVG, FLOORSMIN_AVG,
        LANDAREA_AVG, LIVINGAPARTMENTS_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, COMMONAREA_MODE, LANDAREA_MODE,
        LIVINGAPARTMENTS_MODE,
        BASEMENTAREA_MEDI, TOTALAREA_MODE, FLAG_DOCUMENT_3, NAME_CONTRACT_TYPE_Cash_loans, CODE_GENDER_F, NAME_TYPE_SUITE_Unaccompanied,
        NAME_INCOME_TYPE_Working, EDUCATION_TYPE, FAMILY_STATUS, HOUSING_TYPE, HOUSETYPE_MODE
    ]
    transformed_data = scaler.fit_transform([features])
    prediction = loaded_file.predict(transformed_data)[0]
    return prediction
    
if __name__ == "__main__":
    prediction = main()
    if st.button("predict"):
        if prediction == 0:
            st.write("Credit card was not found defaulting")
        else:
            st.write("Credit card is prone to threat")
