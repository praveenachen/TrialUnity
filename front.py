import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import plotly.express as px
import requests
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(layout = 'wide')

# Function to display the trial information neatly

def display_demographic_distribution(patient_data):
    # Make a copy of the dataset for demographic distribution visualization
    vis_data = patient_data.copy()

    # Proceed with encoding within this copied dataset
    vis_data['Sex'] = vis_data['Sex'].map({0: 'Male', 1: 'Female'})
    vis_data['Race'] = vis_data['Race'].map({
        0: 'White', 1: 'East Asian', 2: 'South Asian', 3: 'Black', 
        4: 'Native American', 5: 'Islander', 6: 'Middle Eastern'
    })
    vis_data['Socioeconomic Status'] = pd.cut(vis_data['Socioeconomic Status'], 
                                              bins=[0, 0.33333, 0.66666, 1], 
                                              labels=['Lower Class', 'Middle Class', 'Upper Class'], 
                                              right=False)
    
    # Layout settings common to all pie charts
    pie_chart_layout = {
        'margin': dict(t=20, b=0, l=0, r=0),
        'width': 300,
        'height': 300,
        'legend_orientation': 'h',
        'legend_x': 0.5,
        'legend_y': -0.15,
        'legend_xanchor': 'center',
        'legend_title_text': '',
        'uniformtext_minsize': 10,
        'uniformtext_mode': 'hide'
    }

    # Create three columns for the pie charts with some spacing
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # Generate and display pie chart for Sex distribution
    with col1:
        fig_sex = px.pie(vis_data, names='Sex', title='Sex Distribution')
        fig_sex.update_layout(**pie_chart_layout)
        st.plotly_chart(fig_sex, use_container_width=True)

    # Generate and display pie chart for Race distribution
    with col2:
        fig_race = px.pie(vis_data, names='Race', title='Race Distribution')
        fig_race.update_layout(**pie_chart_layout)
        st.plotly_chart(fig_race, use_container_width=True)

    # Generate and display pie chart for SES distribution
    with col3:
        fig_ses = px.pie(vis_data, names='Socioeconomic Status', title='SES Distribution')
        fig_ses.update_layout(**pie_chart_layout)
        st.plotly_chart(fig_ses, use_container_width=True)

# This function is expected to be called within your Streamlit app code
# display_demographic_distribution(patient_data)  # assuming patient_data is defined

def calculate_additional_recruitment(selected_patients):
    # Initialize the dictionary to hold recruitment needs
    needed_recruitment = {}
    
    # Define your category encodings
    sex_encoding = {0: 'Male', 1: 'Female'}
    race_encoding = {0: 'White', 1: 'East Asian', 2: 'South Asian', 3: 'Black', 4: 'Native American', 5: 'Islander', 6: 'Middle Eastern'}

    # Define the target number for each group, which is the total count divided by the number of unique groups
    target_counts = {
        'Sex': len(selected_patients) / selected_patients['Sex'].nunique(),
        'Race': len(selected_patients) / selected_patients['Race'].nunique(),
        'SES Category': len(selected_patients) / selected_patients['SES Category'].nunique()
    }

    # Calculate required recruitment for each category
    for category, target_count in target_counts.items():
        group_counts = selected_patients[category].value_counts().to_dict()
        needed_recruitment[category] = {}
        
        for group, count in group_counts.items():
            # Calculate how many are needed to reach the target
            additional_needed = max(0, int(target_count) - count)
            if additional_needed > 0:
                # Map the group number to the encoding
                if category == 'Sex':
                    group_name = sex_encoding[group]
                elif category == 'Race':
                    group_name = race_encoding[group]
                else:
                    group_name = group
                needed_recruitment[category][group_name] = additional_needed

    # Filter out any categories that do not require additional recruitment
    needed_recruitment = {
        category: {
            group_name: count
            for group_name, count in groups.items() if count > 0
        } for category, groups in needed_recruitment.items()
    }
    
    return needed_recruitment

def process_dataset(patient_data):
    figs = []
    needed_recruitment = {
        'Sex': {}, 
        'Race': {}, 
        'SES Category': {}
    }

    temp_data = patient_data.copy()
    temp_data = temp_data.rename(columns={'Socioeconomic Status': 'SES'})
    patient_data = patient_data[patient_data['Sex'] < 2]

    # Standardizing features for clustering
    features = patient_data[['Sex', 'Race', 'Socioeconomic Status']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Visualizing the distribution of 'Socioeconomic Status'
    col1, col2, col3= st.columns([1, 1, 1])  # The numbers inside the list can be adjusted to change the width ratio of the columns

    with col1:
        # Visualizing the distribution of 'Socioeconomic Status'
        fig_ses, ax_ses = plt.subplots(figsize=(4, 3))  # Adjust the figsize as needed
        sns.histplot(patient_data['Socioeconomic Status'], kde=True, ax=ax_ses)
        ax_ses.set_title('Distribution of SES')
        st.pyplot(fig_ses)

    with col2:
        # Correlation matrix between features
        fig_corr, ax_corr = plt.subplots(figsize=(4, 3))  # Adjust the figsize as needed
        sns.heatmap(temp_data[['Sex', 'Race', 'SES']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Correlation Matrix')
        st.pyplot(fig_corr)
    
    with col3:
        # Plot silhouette scores
        silhouette_scores = []
        range_of_clusters = range(2, 11)
        for k in range_of_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            score = silhouette_score(features_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        fig_silhouette, ax_silhouette = plt.subplots(figsize=(4, 3))
        ax_silhouette.plot(range_of_clusters, silhouette_scores, marker='o', linestyle='-', color='b')
        ax_silhouette.set_xlabel('Number of Clusters')
        ax_silhouette.set_ylabel('Silhouette Score')
        ax_silhouette.set_title('Silhouette Scores for Different Number of Clusters')
        ax_silhouette.grid(True)
        st.pyplot(fig_silhouette)

    # Finding the optimal number of clusters using silhouette scores
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    # Clustering with the optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans_optimal.fit(features_scaled)

    # Assign clusters to each patient
    patient_data['Cluster'] = kmeans_optimal.labels_

    # Calculate representation scores based on cluster sizes
    cluster_sizes = patient_data['Cluster'].value_counts().sort_index()
    representation_scores = 1 / cluster_sizes
    representation_scores_normalized = MinMaxScaler(feature_range=(1, 100)).fit_transform(representation_scores.values.reshape(-1, 1)).flatten()

    # Mapping each cluster to its representation score
    cluster_to_score = {cluster: score for cluster, score in enumerate(representation_scores_normalized)}

    # Assigning ESR Score to each patient
    patient_data['ESR Score'] = patient_data['Cluster'].apply(lambda x: cluster_to_score[x])

    # Categorize 'Socioeconomic Status' into classes
    def categorize_ses(value):
        if value < 0.33333:
            return 'Lower Class'
        elif value <= 0.66666:
            return 'Middle Class'
        else:
            return 'Upper Class'

    patient_data['SES Category'] = patient_data['Socioeconomic Status'].apply(categorize_ses)

    # Redefine strata to include 'SES Category', 'Sex', and 'Race'
    patient_data['Strata'] = patient_data.apply(lambda row: f"{row['Sex']}-{row['Race']}-{row['SES Category']}", axis=1)

    # Calculate selections per stratum for stratified sampling
    strata = patient_data['Strata'].unique()
    selected_indices = []
    n_select = 50
    strata_sizes = patient_data['Strata'].value_counts()
    proportions = strata_sizes / len(patient_data)
    strata_selections = np.floor(proportions * n_select).astype(int)

    # Adjust to ensure the total selected is n_select
    while strata_selections.sum() < n_select:
        strata_selections[strata_selections.idxmin()] += 1

    for stratum in strata:
        stratum_patients = patient_data[patient_data['Strata'] == stratum]
        
        # Stratum size check and weighted selection
        if len(stratum_patients) <= strata_selections[stratum]:
            selected_indices.extend(stratum_patients.index.tolist())
        else:
            weights = stratum_patients['ESR Score'] / stratum_patients['ESR Score'].sum()
            selected = np.random.choice(stratum_patients.index, size=strata_selections[stratum], replace=False, p=weights)
            selected_indices.extend(selected)

    selected_indices = list(set(selected_indices))
    selected_patients = patient_data.loc[selected_indices]
    selected_patients.reset_index(drop=True, inplace=True)

    # Insert this in process_dataset after selected_patients is defined
    needed_recruitment = calculate_additional_recruitment(selected_patients)


    # Display the selected patient details
    # print(selected_patients[['Patient ID', 'Sex', 'Race', 'Socioeconomic Status', 'SES Category', 'ESR Score']])

    return selected_patients, figs, needed_recruitment

def display_additional_recruitment_needed(needed_recruitment):
    for category, needs in needed_recruitment.items():
        if needs:  # Only display if there are needs in the category
            with st.expander(f"{category} Recruitment Needs"):
                max_count = max(needs.values())  # Get the maximum value for normalization
                for group, count in needs.items():
                    if count > 0:  # Display only if additional participants are needed
                        st.write(f"{group}: {count} more participants needed")
                        # Normalize progress bar value to be between 0.0 and 1.0
                        progress_value = count / max_count  # Adjust calculation here
                        st.progress(progress_value)

def hide_uploader_label():
    st.markdown("""
    <style>
    /* Hide the label of the file uploader */
    div[data-baseweb="file-uploader"] > div:first-child > label {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

def physician_dashboard():
    # Column for logo
    col1, col2 = st.columns([1, 6])

    with col1:
        # Open the image file
        image = Image.open("C:\TrialUnity\logo2.png")
        # Display the image
        st.image(image, width = 180)  # You can adjust the width to fit your layout
    
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 5.1em; line-height: 1.5;'>
                <span style='color: #183172;'>Trial</span>
                <span style='color: #1B9CE4;'>Unity</span>
                <span style='color: black;'> | Physician Dashboard</span>
            </h1>
            """, unsafe_allow_html=True)
    st.markdown("----")

    col3, col4, col5, col6 = st.columns([1, 1, 1, 1])

    with col3: 
        image_path = r"C:\TrialUnity\total_patients.png"  # Make sure to use a raw string here
        first_pic = Image.open(image_path)
        st.image(first_pic, use_column_width=True)

    with col4:
        image_path = r"C:\TrialUnity\current_patients.png"  # Make sure to use a raw string here
        second_pic = Image.open(image_path)
        st.image(second_pic, use_column_width=True)
    
    with col5: 
        image_path = r"C:\TrialUnity\total_views.png"  # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)
    
    with col6: 
        image_path = r"C:\TrialUnity\discharges.png"  # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)
    
    col7 = st.columns([1])
    with col7[0]: 
        image_path = r"C:\TrialUnity\enrollment.png"  # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)
    
    # Button to view trial information
    st.markdown("----")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Upload & Analyze New Patient Data
        </h2>
        """, unsafe_allow_html=True)
    hide_uploader_label()  # Call this to hide the file uploader label
    uploaded_file = st.file_uploader("", type=['csv'])  # Label is now redundant and can be empty
    st.markdown("\n")
    st.markdown("\n")

    if uploaded_file is not None:
        patient_data = pd.read_csv(uploaded_file)

        display_demographic_distribution(patient_data)
        # Process the dataset and receive processed data, figures, and findings
        selected_patients, figs, needed_recruitment = process_dataset(patient_data)

        # Display each figure
        for fig in figs:
            st.pyplot(fig)
        
        st.markdown("----")
        st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Updated Patient List
        </h2>
        """, unsafe_allow_html=True)
        st.markdown("\n")
        st.markdown("\n")
        st.dataframe(selected_patients, use_container_width=True)

        st.markdown("----")
        st.markdown("""
                <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
                    Recruitment Reccomendations & Feedback
                </h2>
                """, unsafe_allow_html=True)
        st.markdown("\n")
        st.markdown("\n")
        # Display findings about additional recruitment needed
        display_additional_recruitment_needed(needed_recruitment)  # Pass needed_recruitment here


def fetch_trials(condition, location, intervention, sex, phase):
    base_url = 'https://clinicaltrials.gov/api/query/full_studies?'

    # Construct expression
    expr_parts = [f"cond={condition}"]
    if location != "Prefer not to say":
        expr_parts.append(f"locn={location}")
    if intervention != "No preference":
        expr_parts.append(f"int={intervention}")
    if sex != "Prefer not to say" and sex != "All":
        expr_parts.append(f"sex={sex}")
    if phase != "No preference":
        expr_parts.append(f"phase={phase}")

    # Include statuses that indicate the trial is ongoing or hasn't started yet
    status_filter = "recr=Open"  # 'Open' generally captures ongoing and upcoming trials

    # Complete expression
    expr = "&".join(expr_parts) + "&" + status_filter
    url = f"{base_url}{expr}&min_rnk=1&max_rnk=50&fmt=json"

    # Send request
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        studies = data['FullStudiesResponse']['FullStudies']
        if studies:
            # Process studies into a DataFrame
            results = []
            for study in studies:
                contact_info = study.get('Study', {}).get('ProtocolSection', {}).get('ContactsLocationsModule', {})
                central_contacts = contact_info.get('CentralContactList', {}).get('CentralContact', [])
                
                # Handle both single and multiple contacts
                if isinstance(central_contacts, list):
                    central_contact = central_contacts[0] if central_contacts else {}
                else:
                    central_contact = central_contacts

                results.append({
                    'NCTId': study['Study']['ProtocolSection']['IdentificationModule']['NCTId'],
                    'Title': study['Study']['ProtocolSection']['IdentificationModule']['BriefTitle'],
                    'Status': study['Study']['ProtocolSection']['StatusModule']['OverallStatus'],
                    'Contact Name': central_contact.get('CentralContactName', 'N/A'),
                    'Contact Phone': central_contact.get('CentralContactPhone', 'N/A'),
                    'Contact Email': central_contact.get('CentralContactEMail', 'N/A')
                })
            return pd.DataFrame(results)
        else:
            st.write("No studies found. Query used:", url)
            return pd.DataFrame()
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        st.write("Response error:", response.text)
        st.write("Query used:", url)
        return pd.DataFrame()

def patient_dashboard():
    col1, col2 = st.columns([1, 6])

    # Column for logo
    with col1:
        # Open the image file
        image = Image.open("C:\TrialUnity\logo2.png")
        # Display the image
        st.image(image, width = 180)  # You can adjust the width to fit your layout
    
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 5.5em; line-height: 1.5;'>
                <span style='color: #183172;'>Trial</span>
                <span style='color: #1B9CE4;'>Unity</span>
                <span style='color: black;'> | Patient Dashboard</span>
            </h1>
            """, unsafe_allow_html=True)
    
    st.markdown("""---""")
    col3 = st.columns([1])
    with col3[0]: 
        image_path = r"C:\TrialUnity\banner.png"  # Make sure to use a raw string here
        first_pic = Image.open(image_path)
        st.image(first_pic, use_column_width=True)
    st.markdown("\n")
    condition = st.text_input("Enter the disease condition to find clinical trials:", value="Cancer")
    location = st.text_input("Enter the location (city or country):", "Prefer not to say")
    intervention = st.text_input("Enter preferred intervention/treatment:", "No preference")
    sex = st.selectbox("Select sex:", options=["Prefer not to say", "Male", "Female", "All"])
    phase = st.selectbox("Select study phase:", options=["No preference", "Phase 1", "Phase 2", "Phase 3", "Phase 4"])

    if st.button('Search'):
        df_trials = fetch_trials(condition, location, intervention, sex, phase)
        st.markdown("""---""")
        if not df_trials.empty:
            st.dataframe(df_trials)
        else:
            st.write("No trials found for the selected criteria.")

def home_page():
    # Title and logo side by side
    col1, col2 = st.columns([1, 5])

    # Column for logo
    with col1:
        # Open the image file
        image = Image.open("C:\TrialUnity\logo2.png")
        # Display the image
        st.image(image, width = 200)  # You can adjust the width to fit your layout

    # Column for title
    with col2:
        st.markdown("""
            <h1 style='text-align: left; margin-bottom: 0; padding-bottom: 0; font-size: 7em; line-height: 1.5;'>
                <span style='color: #183172;'>Trial</span>
                <span style='color: #1B9CE4;'>Unity</span>
                <span style='color: black;'> | About Us</span>
            </h1>
            """, unsafe_allow_html=True)
        
    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            <i>Prioritizing Diversity, Advancing Health<i>
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    
    col3, col4, col5 = st.columns([1, 1, 1])

    with col3: 
        image_path = r"C:\TrialUnity\pic1.png"  # Make sure to use a raw string here
        first_pic = Image.open(image_path)
        st.image(first_pic, use_column_width=True)

    with col4:
        image_path = r"C:\TrialUnity\pic2.png"  # Make sure to use a raw string here
        second_pic = Image.open(image_path)
        st.image(second_pic, use_column_width=True)
    
    with col5: 
        image_path = r"C:\TrialUnity\pic3.png"  # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)

    st.markdown("""
    <div style='font-size: 40px;'>
        We envision a world where <span style='color: #1B9CE4;'>every patient</span> can access 
        <span style='color: #1B9CE4;'>cutting-edge medical care</span> and contribute to 
        <span style='color: #1B9CE4;'>scientific advancements</span>, moving us closer to a 
        <span style='color: #1B9CE4;'>healthcare system</span> that <span style='color: #1B9CE4;'>benefits everyone</span>, 
        without exception.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""---""")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Our Mission
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("""
        <div style='font-size: 30px;'>
            At Trial Unity, our mission is to transform the landscape of clinical trials by ensuring they are more inclusive. We strive to dismantle historical biases in medical research by ensuring that every patient, especially those from underrepresented groups, has equal access to participate in clinical trials. Our goals directly align with adressing the United Nations SDG's of Good Health & Well-being and Reduced Inequalities. 
        </div>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Status Quo
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("""
        <div style='font-size: 30px;'>
        The increasing number of global clinical trials, almost 30 thousand annually, holds promise for scientific breakthroughs, but the persisting issue of inadequate diversity in participant representation poses a significant challenge. Despite comprising around 40% of the U.S. population, minority groups are vastly underrepresented, with White participants dominating FDA-approved drug trials at 75% in 2020.         </div>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    blue_gradient_colors = ['#1B9CE4', '#183172', '#54AFE6', '#183E9F']

    # Data for the pie charts
    sex_data = ['Female', 'Male']
    sex_values = [51, 49]

    race_data = ['White', 'Asian', 'Black or African American', 'Other', 'American Indian or Alaska Native']
    race_values = [76, 11, 7, 5, 1]

    ethnicity_data = ['Hispanic or Latino', 'Not Hispanic or Latino', 'Missing']
    ethnicity_values = [20, 67, 13]

    # Create the pie charts with the blue gradient
    fig_sex = go.Figure(go.Pie(labels=sex_data, values=sex_values, marker_colors=blue_gradient_colors[:2]))
    fig_race = go.Figure(go.Pie(labels=race_data, values=race_values, marker_colors=blue_gradient_colors))
    fig_ethnicity = go.Figure(go.Pie(labels=ethnicity_data, values=ethnicity_values, marker_colors=blue_gradient_colors[:3]))

    # Customizing the pie charts
    fig_sex.update_traces(hole=.4, hoverinfo="label+percent", textinfo="percent", textposition="inside")
    fig_sex.update_layout(title_text="Sex Distribution", showlegend=True)

    fig_race.update_traces(hole=.4, hoverinfo="label+percent", textinfo="percent", textposition="inside")
    fig_race.update_layout(title_text="Race Distribution", showlegend=True)

    fig_ethnicity.update_traces(hole=.4, hoverinfo="label+percent", textinfo="percent", textposition="inside")
    fig_ethnicity.update_layout(title_text="Ethnicity Distribution", showlegend=True)

    # Creating columns for the pie charts
    col1, col2, col3 = st.columns(3)

    # Displaying each pie chart in its respective column
    with col1:
        st.plotly_chart(fig_sex, use_container_width=True)

    with col2:
        st.plotly_chart(fig_race, use_container_width=True)

    with col3:
        st.plotly_chart(fig_ethnicity, use_container_width=True)
    
    st.markdown("""
        <div style='font-size: 30px;'>
        This discrepancy compromises the generalizability of findings and hinders the development of tailored treatments, ultimately impeding equitable healthcare outcomes. Addressing this requires systemic changes to enhance physician engagement in recruitment and improve inclusivity in trial design, ensuring that medical advancements benefit all demographics equally. Failure to rectify this issue perpetuates healthcare disparities and stifles medical progress, hindering the goal of providing optimal care to diverse patient populations.        """, 
        unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Our Solution
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("""
        <div style='font-size: 30px;'>
        Trial Unity is a web application that connects patients with clinical trials while aiding researchers in selecting participants impartially. Leveraging artificial neural networks, it tailors trial suggestions based on medical needs, eligibility criteria, and proximity, while considering socio-economic status, sex, and race to generate an ESR score for researchers to use in participant selection, aiming to achieve diverse representation. The app keeps patients engaged with real-time notifications, updates, and secure messaging features, promising to revolutionize medical research by promoting inclusivity, patient-friendliness, and efficiency. """, 
        unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("""
        <h2 style='font-family: "Apple Chancery", cursive; color: #183172; text-align: left; font-size: 4em; line-height:0.5;'>
            Meet the Team 
        </h2>
        """, unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")
    col6, col7, col8, col9 = st.columns([1, 1, 1, 1])

    with col6: 
        image_path = r"C:\TrialUnity\praveena.png" # Make sure to use a raw string here
        first_pic = Image.open(image_path)
        st.image(first_pic, use_column_width=True)

    with col7:
        image_path = r"C:\TrialUnity\orianna.png"  # Make sure to use a raw string here
        second_pic = Image.open(image_path)
        st.image(second_pic, use_column_width=True)
    
    with col8: 
        image_path = r"C:\TrialUnity\shreeya.png" # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)
    
    with col9: 
        image_path = r"C:\TrialUnity\leah.png"  # Make sure to use a raw string here
        third_pic = Image.open(image_path)
        st.image(third_pic, use_column_width=True)
    

# Main application routing
def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Go to", ["Home", "Physician Dashboard", "Patient Dashboard"])
    
    if app_page == "Home":
        home_page()
    elif app_page == "Physician Dashboard":
        physician_dashboard()
    elif app_page == "Patient Dashboard":
        patient_dashboard()

if __name__ == "__main__":
    main()