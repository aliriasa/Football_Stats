
#python -m streamlit run script.py
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Load the dataframe
    st.set_page_config(
        page_title="Football Statistics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df = load_dataframe()  # Replace `load_dataframe()` with your code to load the dataframe

    # Add a sidebar for filtering (if needed)
    filtered_df, similarity, selected_player = sidebar_filters(df)

    format_dataframe(filtered_df, similarity, selected_player)

def load_dataframe():
    # Replace with your code to load the dataframe
    # For example:
    df = pd.read_csv('transfer_stats.csv')
    return df

def apply_similarity_formula(model_player_data, row):
    valid_indices = ~np.isnan(model_player_data)
    if len(valid_indices) == 0:
        return 0
    else:
        arr1 = np.array(model_player_data[valid_indices]).reshape(1, -1)
        arr2 = np.array(np.nan_to_num(row[valid_indices], nan=0) ).reshape(1, -1)
        
        similarity = cosine_similarity(arr1, arr2)[0, 0]
        return round(similarity, 4)
    
def sidebar_filters(df):
    # Initialize the filtered dataframe with the original dataframe
    filtered_df = df.copy()
    similarity = False
    selected_player_df = None

    # Add the right sidebar for player selection
    st.sidebar.title('Player Selection')
    player_names = df['Nombre'].unique()
    selected_player = st.sidebar.selectbox('Select Player', ['All'] + list(player_names))

    if selected_player != 'All':
        # Filter for the selected player
        data_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        data_columns.remove('Valor de mercado')
        data_columns.remove('Edad')
        data_columns.remove('Altura')

        model_player_data = filtered_df.loc[filtered_df['Nombre'] == selected_player, data_columns].values[0]

        filtered_df['Similarity'] = filtered_df[data_columns].apply(
            lambda row: apply_similarity_formula(model_player_data, row[data_columns].values),
            axis=1
        )

        # Sort the DataFrame by similarity score
        selected_player_df = filtered_df.loc[filtered_df['Nombre'] == selected_player]
        filtered_df = filtered_df.sort_values(by='Similarity', ascending=False).loc[filtered_df['Nombre'] != selected_player]
        similarity = True


    st.sidebar.title('Data Filters')

    # Get the categorical column names
    categorical_columns = ['Liga', 'Equipo', 'Posicion', 'Pie', 'Lesionado']

    # Get the numerical column names
    numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()

    # Add filter options for categorical and numerical columns
    column_filters = st.sidebar.multiselect('Columns', categorical_columns + numerical_columns)

    for column in column_filters:
        if column in categorical_columns:
            st.sidebar.markdown(f"**Filtering by {column}**")
            # Get unique values for the column, filtered based on previous filters
            column_values = filtered_df[column].loc[filtered_df[column].notnull()]
            unique_values = column_values.unique()
            # Display all possible values as a dropdown menu
            all_option = 'All' in column_values
            if all_option:
                selected_values = st.sidebar.multiselect(column, unique_values, default='All')
                if 'All' in selected_values:
                    filtered_df = filtered_df.drop(columns=[column])
            else:
                selected_values = st.sidebar.multiselect(column, unique_values)
                # Filter the dataframe based on the selected values
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
        else:
            # Apply the filters based on the selected numerical column
            st.sidebar.markdown(f"**Filtering by {column}**")
            min_value = st.sidebar.number_input(f'Minimum Value for {column}', value=filtered_df[column].min())
            max_value = st.sidebar.number_input(f'Maximum Value for {column}', value=filtered_df[column].max())
            filtered_df = filtered_df[(filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)]

    # Return the filtered dataframe
    return filtered_df, similarity, selected_player_df


def format_dataframe(df, similarity, selected_player_df):
    # Create a copy of the dataframe to avoid modifying the original
    formatted_df = pd.concat([df.copy(), selected_player_df], axis=0)
    
    # Define the columns to display initially
    columns_to_display = ['Liga', 'Equipo', 'Nombre', 'Imagen', 'Posicion', 'Nacionalidad', 'Valor de mercado']

    if similarity:
        selected_player = np.array(selected_player_df['Nombre'])[0]
        columns_to_display.append('Similarity')

    # Create an expander for displaying additional columns
    with st.expander("More Columns", expanded=False):
        # Get the list of remaining columns
        remaining_columns = [col for col in formatted_df.columns if col not in columns_to_display]

        # Add checkbox for each remaining column
        selected_columns = st.multiselect("Select columns to display", remaining_columns, default=[])
        columns_to_display.extend(selected_columns)

        # Add a dropdown menu to select a column for sorting
        sort_columns = st.multiselect("Select columns to sort", columns_to_display, default=['Valor de mercado'])

    # Filter the dataframe based on the selected columns
    formatted_df = formatted_df[columns_to_display]

    # Sort the dataframe based on the selected columns
    if sort_columns and not similarity:
        # Sort the dataframe by multiple columns
        sort_orders = [False if pd.api.types.is_numeric_dtype(formatted_df[col].dtype) else True for col in sort_columns]

        # Convert NaN values to appropriate sentinel values for sorting
        sentinel = np.inf if sort_orders[0] else -np.inf
        formatted_df[sort_columns] = formatted_df[sort_columns].fillna(sentinel)

        # Perform multi-column sorting
        formatted_df = formatted_df.sort_values(by=sort_columns, ascending=sort_orders, na_position='last')

        # Restore NaN values in sorted columns
        formatted_df[sort_columns] = formatted_df[sort_columns].replace(sentinel, np.nan)

    # Replace the 'Imagen' column with formatted HTML to display the image
    formatted_df['Imagen'] = formatted_df['Imagen'].apply(lambda url: f'<img src="{url}" alt="Image" width="50px">')

    if similarity:
        # Filter for the selected player and display as a single-row table
        selected_player_table = formatted_df.loc[formatted_df['Nombre'] == selected_player]

        # Filter for other players and display as a table
        other_players_table = formatted_df.loc[formatted_df['Nombre'] != selected_player]

        label_row = pd.DataFrame([[''] * len(columns_to_display)], columns=columns_to_display)

        # Concatenate the DataFrames with the label row
        concatenated_df = pd.concat([selected_player_table, label_row, other_players_table], ignore_index=True).to_html(escape=False, index=False)
        st.write(concatenated_df, unsafe_allow_html=True)

    else:
        # Display the formatted dataframe as a single table
        html_table = formatted_df.to_html(escape=False, index=False)
        st.write(html_table, unsafe_allow_html=True)



if __name__ == '__main__':
    main()


