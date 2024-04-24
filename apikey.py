

def main():
    st.title("Patient Dashboard")

    condition = st.text_input("Enter the disease condition to find clinical trials:", value="Cancer")
    location = st.text_input("Enter the location (city or country):", "Prefer not to say")
    intervention = st.text_input("Enter preferred intervention/treatment:", "No preference")
    sex = st.selectbox("Select sex:", options=["Prefer not to say", "Male", "Female", "All"])
    phase = st.selectbox("Select study phase:", options=["No preference", "Phase 1", "Phase 2", "Phase 3", "Phase 4"])

    if st.button('Search'):
        df_trials = fetch_trials(condition, location, intervention, sex, phase)
        if not df_trials.empty:
            st.dataframe(df_trials)
        else:
            st.write("No trials found for the selected criteria. Debug info provided in console.")

if __name__ == "__main__":
    main()
