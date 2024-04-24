import os
import streamlit as st
import openai

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.title("Patient Dashboard")

    with st.form("patient_query_form"):
        age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)
        condition = st.text_input("What is your medical condition?")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            trials = search_clinical_trials(age, condition)
            st.subheader("Relevant Clinical Trials:")
            st.write(trials)

def search_clinical_trials(age, condition):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Formulate the messages for the chat
    messages = [
        {"role": "system", "content": f"Searching for clinical trials for a {age}-year-old with {condition}."},
        {"role": "user", "content": "What are the most relevant clinical trials?"}
    ]
    
    # Use the chat completion API
    response = client.chat_completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    trials_info = response.choices[0].message['content']  # Adjusted to new response format
    
    return trials_info

if __name__ == "__main__":
    main()
